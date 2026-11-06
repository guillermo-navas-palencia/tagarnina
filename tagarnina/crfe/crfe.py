import time
from numbers import Integral
from typing import Any, Callable, Hashable, Self

import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import check_scoring
from sklearn.utils._param_validation import Interval, RealNotInt

from ..optimization.constraints import Constraints
from .base import BaseCRFE, column_scorer, resolve_n_features
from .history import History
from .logger import CRFELogger
from .monitoring import Monitoring


class CRFE(BaseCRFE):
    """Feature ranking with constrained recursive feature elimination.

    Parameters
    ----------
    estimator : estimator instance
        Supervised estimator, cloned and refit at each iteration.

    n_features_to_select : int, float or None, default=None
        Number of features to select. If None, half of the features are
        selected when `constraints` is None, and the search only stops at one
        feature (or when no further elimination is feasible) otherwise. A float
        in (0, 1] is a fraction of the features. With `constraints`, it is
        added as an at-least-k constraint over all features.

    step : int or float, default=1
        Number of features to eliminate per iteration. A float in (0, 1) is
        the fraction of the features, truncated to at least one.

    constraints : Constraints or None, default=None
        Constraints that every selected subset must satisfy. If None, the
        elimination is the standard RFE.

    strategy : {"greedy", "probing"}, default="probing"
        Only used if `constraints` is not None.

        - "greedy": the `step` least important features are excluded and the
          search stops if this is infeasible.
        - "probing": if excluding the `step` least important features is
          infeasible, candidates are tested one by one and those that cannot
          be excluded are skipped.

    scoring : str, callable or None, default=None
        Scorer, as in scikit-learn (see `sklearn.metrics.check_scoring`),
        evaluated on the training data for each fitted support and recorded
        in `history_`. It is only a record for reporting and plotting: it
        never influences the elimination or the selected features. Because it
        is computed on the training data, it is optimistic and tends to
        increase with the number of features.

    solver_params : dict or None, default=None
        Only used if `constraints` is not None. Parameters of the CP-SAT
        solver: `num_workers` (default 1) and `time_limit` in seconds per
        solver call (default 10). With more than one worker the solutions
        may not be reproducible.

    importance_getter : str or callable, default="auto"
        Attribute name of the fitted estimator holding the importances, or a
        callable called as `getter(estimator=..., X=..., **importance_params)`.
        Attribute importances are squared as in RFE, and callable outputs are
        made absolute.

    importance_params : dict or None, default=None
        Extra keyword arguments passed to a callable `importance_getter`.

    importance_offset : float, default=1e-3
        Only used if `constraints` is not None. The importances of the active
        features are normalized to a maximum of 1 and this offset is added
        before they enter the solver objective, and must be in [0, 0.1]. With a
        positive offset, features with zero importance are kept unless the
        constraints force their removal, instead of being dropped by an
        arbitrary tie-break. Features with normalized importance below the
        offset are effectively tied with zero importance. The ranking of
        the `step` least important features does not depend on it.

    callback : callable or None, default=None
        Called as `callback(row)` after each fitted support, including the
        final one. `row` is a dict with the `history_` fields of that fit
        (`n_features`, `support`, `score`, `fit_time`, `solve_time`,
        `elapsed`, ...) plus `iteration`. If it returns a truthy value the
        fit stops and the support of that row is selected (the return value
        is ignored for the final one). It is called between fits, so it cannot
        interrupt a running fit. Exceptions propagate. Lambdas prevent the
        selector from being pickled.

    verbose : int, default=0
        0 shows nothing, 1 prints the settings, one line per fitted support and
        a summary, and 2 also prints the solver and model details.

    Attributes
    ----------
    support_ : ndarray of shape (n_features,)
        Mask of the selected features.

    ranking_ : ndarray of shape (n_features,)
        Elimination ranking, 1 for the selected features.

    history_ : dict of ndarray
        One entry per fitted support, from the full set to the selected one:
        `n_features`, `support` (boolean matrix of shape (n_rounds,
        n_features)), `retained_importance`, `importance_gap` and, if
        `scoring` is set, `score`. `retained_importance` is the fraction of
        the importance of the support kept by the next elimination.
        `importance_gap` is how much more of it would be kept by the same
        number of most important features, as RFE would, so it measures the
        importance cost of the constraints (0 without constraints). Both
        are NaN for the last support. The mean importance of the kept
        features follows from `retained_importance` and the next
        `n_features`. `removed` is the number of features removed by that
        elimination, `n_feasible_probes` and `n_infeasible_probes` count its
        probes by outcome (zero unless `strategy` is "probing"), and
        `fit_time`, `solve_time` and `elapsed` are in seconds (`elapsed` is
        cumulative). `removed`, the probe counts and `solve_time` are NaN for
        the last support.

    summary_ : dict
        `termination` reason (a `Termination` member), `wall_time` and
        `cpu_time` of the fit, the time spent fitting (`fit_time`), solving
        (`solve_time`) and scoring (`score_time`), the probes of the probing
        strategy by outcome (`n_feasible_probes`, `n_infeasible_probes`), and
        `n_blocked` features that could not be removed.
    """

    _parameter_constraints: dict = {
        **BaseCRFE._parameter_constraints,
        "n_features_to_select": [
            None,
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="right"),
        ],
        "callback": [callable, None],
        "scoring": [str, callable, None],
    }

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_features_to_select: int | float | None = None,
        step: int | float = 1,
        constraints: Constraints | None = None,
        strategy: str = "probing",
        solver_params: dict[str, Any] | None = None,
        scoring: str | Callable | None = None,
        importance_getter: str | Callable = "auto",
        importance_params: dict[str, Any] | None = None,
        importance_offset: float = 1e-3,
        callback: Callable[[dict[str, Any]], Any] | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.constraints = constraints
        self.strategy = strategy
        self.solver_params = solver_params
        self.scoring = scoring
        self.importance_getter = importance_getter
        self.importance_params = importance_params
        self.importance_offset = importance_offset
        self.callback = callback
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame | npt.NDArray,
        y: pd.Series | npt.NDArray,
        **fit_params: dict[Hashable, Any],
    ) -> Self:
        """Fit the selector and select the features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. A DataFrame must have string column names. With
            `constraints`, constraints that refer to features by name require
            a DataFrame, otherwise features are named `x0, x1, ...`.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        **fit_params : dict
            Additional parameters passed to the `fit` method of `estimator`.

        Returns
        -------
        self : CRFE
            Fitted selector.

        Raises
        ------
        ValueError
            If a parameter is invalid, or if `constraints` are infeasible.
        """
        self._validate_params()

        start = time.perf_counter()
        cpu_start = time.process_time()

        X_rfe, X, y, feature_names, non_numeric_features = self._check_input(X=X, y=y)
        n_features = X.shape[1]

        scorer = (
            None
            if self.scoring is None
            else column_scorer(
                check_scoring(self.estimator, scoring=self.scoring), X=X_rfe, y=y
            )
        )

        monitor = Monitoring(
            scorer=scorer,
            history=History(scored=scorer is not None),
            log=CRFELogger(
                verbose=self.verbose,
                scored=scorer is not None,
                solver=self.constraints is not None,
                probing=self.constraints is not None and self.strategy == "probing",
            ),
            callback=self.callback,
            start=start,
        )

        if self.constraints is not None and self.n_features_to_select is None:
            n_select = 1
        else:
            n_select = resolve_n_features(
                n_features_to_select=self.n_features_to_select, n_features=n_features
            )

        build_time = self._setup(
            X=X,
            feature_names=feature_names,
            non_numeric_features=non_numeric_features,
            n_select=n_select,
        )
        self._print_header(
            monitor=monitor, X=X_rfe, n_select=n_select, build_time=build_time
        )

        support, ranking, termination = self._run(
            X=X_rfe,
            y=y,
            feature_names=feature_names,
            n_select=n_select,
            monitor=monitor,
            fit_params=fit_params,
        )

        self._set_selected_features(
            X=X_rfe,
            y=y,
            support=support,
            ranking=ranking,
            monitor=monitor,
            fit_params=fit_params,
        )

        self._summarize(monitor=monitor, termination=termination, cpu_start=cpu_start)

        return self
