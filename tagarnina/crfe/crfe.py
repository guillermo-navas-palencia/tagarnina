import numbers
import time
from dataclasses import dataclass
from numbers import Integral, Real
from operator import attrgetter
from typing import Any, Callable, Hashable, Self

import numpy as np
import numpy.typing as npt
import ortools
import pandas as pd
from scipy.sparse import issparse

from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone
)
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring

from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    RealNotInt,
    StrOptions,
    validate_parameter_constraints
)
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data
)

from ..optimization.constraints import AtLeastKConstraint, Constraints
from ._log import FitLog, describe_constraints, verbose_logging
from .solver import CRFESolver


def _select_columns(
    X: pd.DataFrame | npt.NDArray,
    *,
    columns: npt.NDArray
) -> pd.DataFrame | npt.NDArray:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, columns]
    return X[:, columns]


def _importance_stats(
    importances: npt.NDArray,
    *,
    kept: npt.NDArray
) -> tuple[float, float]:
    """Fraction of the importance kept, and its shortfall with respect to
    keeping the same number of most important features (as RFE would)."""
    total = importances.sum()
    if not total > 0:
        return np.nan, np.nan

    retained = importances[kept].sum() / total
    best = np.sort(importances)[::-1][:kept.sum()].sum() / total
    return float(retained), float(max(best - retained, 0.0))


class _History:
    """Per-round record of the supports fitted by the elimination loop."""

    _FLOATS = (
        "removed",
        "n_feasible_probes",
        "n_infeasible_probes",
        "retained_importance",
        "importance_gap",
        "fit_time",
        "solve_time",
        "elapsed",
    )

    def __init__(self, *, scored: bool):
        self.scored = scored
        self.rows = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def to_dict(self) -> dict[str, npt.NDArray]:
        keys = [*self._FLOATS, "score"] if self.scored else self._FLOATS

        history = {
            "n_features": np.array([r["n_features"] for r in self.rows], dtype=int),
            "support": np.array([r["support"] for r in self.rows], dtype=bool),
        }
        for key in keys:
            history[key] = np.array([r[key] for r in self.rows], dtype=float)
        return history


@dataclass(kw_only=True)
class _Run:
    """State of a fit: history, log and time spent in each step."""

    scorer: Callable | None
    history: _History
    log: FitLog
    start: float
    fit_time: float = 0.0
    solve_time: float = 0.0
    score_time: float = 0.0
    n_feasible_probes: int = 0
    n_infeasible_probes: int = 0

    def fit(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame | npt.NDArray,
        y: npt.NDArray,
        /,
        **fit_params: Any
    ) -> float:
        start = time.perf_counter()
        estimator.fit(X, y, **fit_params)
        elapsed = time.perf_counter() - start
        self.fit_time += elapsed
        return elapsed

    def score(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame | npt.NDArray,
        y: npt.NDArray
    ) -> float:
        if self.scorer is None:
            return np.nan

        start = time.perf_counter()
        score = float(self.scorer(estimator, X, y))
        self.score_time += time.perf_counter() - start
        return score

    def record(self, row: dict[str, Any]) -> None:
        row = {**row, "elapsed": time.perf_counter() - self.start}
        self.history.append(row)
        self.log.row(row)


class CRFE(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
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

    strategy : {"greedy", "probing"}, default="greedy"
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
        positive offset, features
        with zero importance are kept unless the constraints force their
        removal, instead of being dropped by an arbitrary tie-break. Features
        with normalized importance below the offset are effectively tied with
        zero importance. The ranking of the `step` least important features
        does not depend on it.

    verbose : int, default=0
        0 shows nothing, 1 shows the log of the fit (settings, one line per
        fitted support and a summary) and 2 adds the solver and model details.
        The output is produced with the `logging` module (logger
        `tagarnina`), so it can also be configured without `verbose`.

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
        `termination` reason, `wall_time` and `cpu_time` of the fit, the time
        spent fitting (`fit_time`), solving (`solve_time`) and scoring
        (`score_time`), the probes of the probing strategy by outcome
        (`n_feasible_probes`, `n_infeasible_probes`), and `n_blocked`
        features that could not be removed.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_features_to_select": [
            None,
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="right"),
        ],
        "step": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
        ],
        "constraints": [Constraints, None],
        "strategy": [StrOptions({"greedy", "probing"})],
        "solver_params": [dict, None],
        "scoring": [str, callable, None],
        "importance_getter": [str, callable],
        "importance_params": [dict, None],
        "importance_offset": [Interval(Real, 0, 0.1, closed="both")],
        "verbose": ["verbose"],
    }

    _solver_params_constraints: dict = {
        "num_workers": [Interval(Integral, 1, None, closed="left")],
        "time_limit": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_features_to_select: int | float | None = None,
        step: int | float = 1,
        constraints: Constraints | None = None,
        strategy: str = "greedy",
        solver_params: dict[str, Any] | None = None,
        scoring: str | Callable | None = None,
        importance_getter: str | Callable = "auto",
        importance_params: dict[str, Any] | None = None,
        importance_offset: float = 1e-3,
        verbose: int = 0
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
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame | npt.NDArray,
        y: pd.Series | npt.NDArray,
        **fit_params: dict[Hashable, Any]
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

        with verbose_logging(self.verbose):
            self._fit(X, y, **fit_params)

        return self

    def _fit(
        self,
        X: pd.DataFrame | npt.NDArray,
        y: pd.Series | npt.NDArray,
        **fit_params: dict[Hashable, Any]
    ) -> None:
        start = time.perf_counter()
        cpu_start = time.process_time()

        feature_names = None
        non_numeric_features = frozenset()

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            if not all(isinstance(name, str) for name in feature_names):
                raise TypeError("DataFrame column names must all be strings.")

            non_numeric_features = frozenset(
                name for name, dtype in X.dtypes.items()
                if not pd.api.types.is_numeric_dtype(dtype)
            )

        scorer = (
            None
            if self.scoring is None
            else check_scoring(self.estimator, scoring=self.scoring)
        )
        run = _Run(
            scorer=scorer,
            history=_History(scored=scorer is not None),
            log=FitLog(
                scored=scorer is not None,
                solver=self.constraints is not None,
                probing=self.constraints is not None and self.strategy == "probing"
            ),
            start=start
        )

        X_init = X
        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            accept_sparse="csc",
            dtype=None,
            ensure_min_features=2,
            ensure_all_finite=False,
            multi_output=True,
        )

        if self.constraints is None:
            self.solver_ = None

            if isinstance(X_init, pd.DataFrame):
                X_rfe = X_init
            else:
                X_rfe = X

            termination = self._rfe(X=X_rfe, y=y, run=run, **fit_params)
        else:
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(self.n_features_in_)]

            self.constraints.validate(
                features=feature_names,
                non_numeric_features=non_numeric_features
            )

            n_features = len(feature_names)
            n_select = (
                1
                if self.n_features_to_select is None
                else self._get_n_features_to_select(n_features)
            )

            solver_constraints = Constraints(
                constraints=[
                    *self.constraints.constraints,
                    AtLeastKConstraint(
                        features=feature_names,
                        min_features=n_select
                    ),
                ]
            )

            self.solver_ = CRFESolver(
                features=feature_names,
                constraints=solver_constraints,
                strategy=self.strategy,
                **(self.solver_params or {})
            )

            X_rfe = X_init if isinstance(X_init, pd.DataFrame) else X
            X_dense = X.toarray() if issparse(X) else X
            X_df = pd.DataFrame(data=X_dense, columns=feature_names).infer_objects()

            build_start = time.perf_counter()
            self.solver_.build_constraints(X=X_df)
            build_time = time.perf_counter() - build_start

            params = self.solver_.solver.parameters
            run.log.start(
                [
                    *self._start_lines(X=X_rfe, n_select=n_select),
                    ("Constraints", describe_constraints(self.constraints)),
                ],
                details=[
                    (
                        "Solver",
                        f"CP-SAT {ortools.__version__}, workers={params.num_workers}, "
                        f"time_limit={params.max_time_in_seconds:g}s, "
                        f"seed={params.random_seed}"
                    ),
                    (
                        "Model",
                        f"{self.solver_.n_variables} variables, "
                        f"{self.solver_.n_constraints} constraints, built in "
                        f"{build_time:.3f}s"
                    ),
                ],
            )

            step = self._get_step(n_features)
            support = np.ones(n_features, dtype=bool)
            ranking = np.ones(n_features, dtype=int)

            termination = "reached the target number of features"
            while True:
                features = np.flatnonzero(support)
                n_remove = min(step, len(features) - n_select)
                if n_remove < 1:
                    break

                estimator = clone(self.estimator)
                X_features = _select_columns(X_rfe, columns=features)
                fit_time = run.fit(estimator, X_features, y, **fit_params)

                importances = np.zeros(n_features)
                importances[features] = self._get_importances(
                    estimator=estimator, X=X_features
                )

                ordered = [
                    feature_names[i]
                    for i in features[
                        np.argsort(importances[features], kind="stable")
                    ]
                ]

                # normalized importances plus a small offset, so that the solver
                # keeps zero-importance features unless constraints force removal
                weights = np.zeros(n_features)
                top = importances[features].max()
                weights[features] = (
                    importances[features] / top if top > 0 else 0.0
                ) + self.importance_offset

                solve_start = time.perf_counter()
                result = self.solver_.eliminate(
                    weights=weights,
                    ordered=ordered,
                    n_remove=n_remove
                )
                solve_time = time.perf_counter() - solve_start
                run.solve_time += solve_time
                run.n_feasible_probes += result.n_feasible_probes
                run.n_infeasible_probes += result.n_infeasible_probes

                if result.status == "INFEASIBLE":
                    termination = "no feasible elimination under the constraints"
                    break
                if result.status not in ("OPTIMAL", "FEASIBLE"):
                    raise RuntimeError(
                        f"{type(self).__name__}: the solver returned "
                        f"{result.status}."
                    )

                solution = result.solution
                candidates = result.candidates

                removed = np.flatnonzero(support & ~solution)
                for index in removed:
                    if feature_names[index] in candidates:
                        continue
                    self.solver_.build_exclude_constraint(
                        feature=feature_names[index]
                    )

                retained, gap = _importance_stats(
                    importances, kept=support & solution
                )
                run.record({
                    "n_features": int(support.sum()),
                    "support": support.copy(),
                    "removed": int(support.sum() - (support & solution).sum()),
                    "n_feasible_probes": result.n_feasible_probes,
                    "n_infeasible_probes": result.n_infeasible_probes,
                    "retained_importance": retained,
                    "importance_gap": gap,
                    "fit_time": fit_time,
                    "solve_time": solve_time,
                    "score": run.score(estimator, X_features, y),
                })

                support = support & solution
                ranking[~support] += 1

            self._set_selected_features(
                X=X_rfe,
                y=y,
                support=support,
                ranking=ranking,
                run=run,
                fit_params=fit_params
            )

        self._summarize(
            run=run,
            termination=termination,
            start=start,
            cpu_start=cpu_start
        )

    def _rfe(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        run: _Run,
        **fit_params: dict[Hashable, Any]
    ) -> str:
        n_features = X.shape[1]
        step = self._get_step(n_features)

        n_features_to_select = self._get_n_features_to_select(n_features)
        run.log.start(self._start_lines(X=X, n_select=n_features_to_select))

        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)

        while support.sum() > n_features_to_select:
            features = np.flatnonzero(support)
            estimator = clone(self.estimator)
            X_features = _select_columns(X, columns=features)
            fit_time = run.fit(estimator, X_features, y, **fit_params)

            # Get importance and rank
            importances = self._get_importances(estimator=estimator, X=X_features)

            ranks = np.argsort(importances, kind="stable")

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, support.sum() - n_features_to_select)

            kept = np.ones(len(features), dtype=bool)
            kept[ranks[:threshold]] = False
            retained, gap = _importance_stats(importances, kept=kept)
            run.record({
                "n_features": int(support.sum()),
                "support": support.copy(),
                "removed": threshold,
                "n_feasible_probes": 0,
                "n_infeasible_probes": 0,
                "retained_importance": retained,
                "importance_gap": gap,
                "fit_time": fit_time,
                "solve_time": 0.0,
                "score": run.score(estimator, X_features, y),
            })

            support[features[ranks[:threshold]]] = False
            ranking[~support] += 1

        self._set_selected_features(
            X=X,
            y=y,
            support=support,
            ranking=ranking,
            run=run,
            fit_params=fit_params
        )
        return "reached the target number of features"

    def _start_lines(
        self,
        *,
        X: pd.DataFrame | npt.NDArray,
        n_select: int
    ) -> list[tuple[str, str]]:
        settings = [f"step={self.step}", f"n_features_to_select={n_select}"]
        if self.constraints is not None:
            settings.append(f"strategy={self.strategy}")
        if self.scoring is not None:
            scoring = self.scoring if isinstance(self.scoring, str) else "callable"
            settings.append(f"scoring={scoring}")

        return [
            ("Estimator", type(self.estimator).__name__),
            ("Data", f"{X.shape[0]} samples, {X.shape[1]} features"),
            ("Settings", ", ".join(settings)),
        ]

    def _summarize(
        self,
        *,
        run: _Run,
        termination: str,
        start: float,
        cpu_start: float
    ) -> None:
        wall_time = time.perf_counter() - start
        cpu_time = time.process_time() - cpu_start
        n_blocked = 0 if self.solver_ is None else len(self.solver_.blocked)

        self.summary_ = {
            "termination": termination,
            "wall_time": wall_time,
            "cpu_time": cpu_time,
            "fit_time": run.fit_time,
            "solve_time": run.solve_time,
            "score_time": run.score_time,
            "n_feasible_probes": run.n_feasible_probes,
            "n_infeasible_probes": run.n_infeasible_probes,
            "n_blocked": n_blocked,
        }

        times = [f"fit {run.fit_time:.3f}"]
        if self.solver_ is not None:
            times.append(f"solve {run.solve_time:.3f}")
        if run.scorer is not None:
            times.append(f"score {run.score_time:.3f}")

        lines = [
            ("Termination", termination),
            (
                "Selected",
                f"{int(self.n_features_)} of {self.n_features_in_} features"
            ),
        ]
        if self.solver_ is not None and self.strategy == "probing":
            lines.append((
                "Probes",
                f"{run.n_feasible_probes} feasible, "
                f"{run.n_infeasible_probes} infeasible, {n_blocked} blocked"
            ))
        lines.append((
            "Time",
            f"wall {wall_time:.3f}s ({', '.join(times)}), cpu {cpu_time:.3f}s"
        ))
        run.log.end(lines)

    def _get_step(self, n_features: int) -> int:
        if 0.0 < self.step < 1.0:
            return int(max(1, self.step * n_features))
        return int(self.step)

    def _set_selected_features(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        support: npt.NDArray,
        ranking: npt.NDArray,
        run: _Run,
        fit_params: dict[str, Any]
    ) -> None:
        X_selected = _select_columns(X, columns=support)
        self.estimator_ = clone(self.estimator)
        fit_time = run.fit(self.estimator_, X_selected, y, **fit_params)
        run.record({
            "n_features": int(support.sum()),
            "support": support.copy(),
            "removed": np.nan,
            "n_feasible_probes": np.nan,
            "n_infeasible_probes": np.nan,
            "retained_importance": np.nan,
            "importance_gap": np.nan,
            "fit_time": fit_time,
            "solve_time": np.nan,
            "score": run.score(self.estimator_, X_selected, y),
        })
        self.history_ = run.history.to_dict()
        self.n_features_ = support.sum()
        self.support_ = support
        self.ranking_ = ranking

    def _get_n_features_to_select(self, n_features: int) -> int:
        n_select = self.n_features_to_select
        if n_select is None:
            return n_features // 2
        if isinstance(n_select, numbers.Integral):
            if not 0 < n_select <= n_features:
                raise ValueError(
                    f"n_features_to_select must be in (0, {n_features}], "
                    f"got {n_select}."
                )
            return int(n_select)
        if not 0.0 < n_select <= 1.0:
            raise ValueError(
                f"n_features_to_select as a float must be in (0, 1], got {n_select}."
            )
        return max(1, int(n_select * n_features))

    def _get_importances(
        self,
        *,
        estimator: BaseEstimator,
        X: npt.NDArray
    ) -> npt.NDArray:
        """Return one importance per feature.
        
        Notes
        -----
        Callables receive X in the fit input type and their output is made
        absolute. Attribute getters are squared, as in RFE."""
        getter = self.importance_getter
        params = self.importance_params or {}
        if callable(getter):
            return np.abs(
                np.asarray(getter(estimator=estimator, X=X, **params))
            )

        if not isinstance(getter, str):
            raise ValueError(
                f"{type(self).__name__}: importance_getter must be a string, "
                "'auto', or a callable."
            )

        if getter == "auto":
            if hasattr(estimator, "coef_"):
                getter = "coef_"
            elif hasattr(estimator, "feature_importances_"):
                getter = "feature_importances_"
            else:
                raise ValueError(
                    f"{type(self).__name__}: importance_getter='auto' requires the "
                    "estimator to expose coef_ or feature_importances_."
                )

        importances = np.asarray(attrgetter(getter)(estimator)) ** 2

        if importances.ndim == 1:
            return importances
        return importances.sum(axis=0)

    def _get_support_mask(self) -> npt.NDArray:
        check_is_fitted(self, attributes="support_")
        return self.support_

    def _validate_params(self) -> None:
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=type(self).__name__
        )

        params = self.solver_params or {}
        unknown = sorted(set(params) - set(self._solver_params_constraints))
        if unknown:
            raise ValueError(
                f"{type(self).__name__}: unknown solver_params {unknown}, valid "
                f"keys are {sorted(self._solver_params_constraints)}."
            )
        validate_parameter_constraints(
            self._solver_params_constraints,
            params,
            caller_name=f"{type(self).__name__} solver_params"
        )