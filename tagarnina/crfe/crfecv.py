import time
from collections.abc import Iterable
from numbers import Integral
from typing import Any, Callable, Hashable, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, check_cv
from sklearn.utils import _safe_indexing
from sklearn.utils._param_validation import Interval

from ..optimization.constraints import Constraints
from .base import BaseCRFE, column_scorer, resolve_n_features
from .history import History
from .logger import CRFELogger, describe_model, describe_solver
from .monitoring import Monitoring


class CRFECV(BaseCRFE):
    _parameter_constraints: dict = {
        **BaseCRFE._parameter_constraints,
        "min_features_to_select": [Interval(Integral, 1, None, closed="left")],
        "cv": ["cv_object"],
        "scoring": [str, callable, None],
    }

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        min_features_to_select: int = 1,
        step: int | float = 1,
        cv: int
        | BaseCrossValidator
        | Iterable[tuple[npt.NDArray, npt.NDArray]]
        | None = None,
        constraints: Constraints | None = None,
        strategy: str = "probing",
        solver_params: dict[str, Any] | None = None,
        scoring: str | Callable | None = None,
        importance_getter: str | Callable = "auto",
        importance_params: dict[str, Any] | None = None,
        importance_offset: float = 1e-3,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.cv = cv
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
        **fit_params: dict[Hashable, Any],
    ) -> Self:
        """Fit the selector, choosing the number of features by cross-validation.

        Each fold runs the elimination on its training rows and scores every
        size on its test rows. The size with the highest mean held-out score
        is selected, and ties go to fewer features. Only the sizes reached by
        every fold are compared, which matters when `constraints` stop some
        folds early. The selector is then refitted on all the data at that
        size.

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
            They are not sliced by fold.

        Returns
        -------
        self : CRFECV
            Fitted selector. `cv_results_` holds, for each compared size, the
            `n_features`, `mean_test_score`, `std_test_score` and
            `split{i}_test_score` keys. `history_` holds the mean held-out
            score of each size in `score`, with NaN for sizes that were not
            compared.

        Raises
        ------
        ValueError
            If a parameter or a constraint is invalid.
        """
        self._validate_params()

        start = time.perf_counter()
        cpu_start = time.process_time()

        X_rfe, X, y, feature_names, non_numeric_features = self._check_input(X=X, y=y)
        n_select = resolve_n_features(
            n_features_to_select=self.min_features_to_select, n_features=X.shape[1]
        )

        scorer = check_scoring(self.estimator, scoring=self.scoring)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_folds = cv.get_n_splits(X_rfe, y)
        solver = self.constraints is not None
        probing = solver and self.strategy == "probing"

        log = CRFELogger(
            verbose=self.verbose, scored=False, solver=solver, probing=probing
        )
        log.print_banner()
        log.print_lines(lines=self._describe_input(X=X_rfe, n_select=n_select))
        if solver:
            log.print_lines(
                lines=describe_solver(solver=self._probe_solver()), min_verbose=2
            )
        log.print_lines(lines=[("CV folds", str(n_folds))])

        fold_scores = []
        fold_monitors = []
        for index, (train, test) in enumerate(cv.split(X_rfe, y), start=1):
            X_train_rfe = _safe_indexing(X_rfe, train)
            X_test = _safe_indexing(X_rfe, test)
            y_train = _safe_indexing(y, train)
            y_test = _safe_indexing(y, test)

            build_time = self._setup(
                X=_safe_indexing(X, train),
                feature_names=feature_names,
                non_numeric_features=non_numeric_features,
                n_select=n_select,
            )
            monitor = Monitoring(
                scorer=column_scorer(scorer, X=X_test, y=y_test),
                history=History(scored=True),
                log=CRFELogger(
                    verbose=2 if self.verbose >= 2 else 0,
                    scored=True,
                    solver=solver,
                    probing=probing,
                ),
                start=time.perf_counter(),
            )
            monitor.log.print_fold_start(
                index=index,
                n_folds=n_folds,
                details=(
                    describe_model(solver=self.solver_, build_time=build_time)
                    if solver
                    else []
                ),
            )

            support, _, _ = self._run(
                X=X_train_rfe,
                y=y_train,
                feature_names=feature_names,
                n_select=n_select,
                monitor=monitor,
                fit_params=fit_params,
            )
            self._record_selected(
                X=X_train_rfe,
                y=y_train,
                support=support,
                monitor=monitor,
                fit_params=fit_params,
            )

            history = monitor.history.to_dict()
            fold_score = dict(
                zip(history["n_features"].tolist(), history["score"].tolist())
            )
            fold_scores.append(fold_score)
            fold_monitors.append(monitor)

            best, best_score = max(
                fold_score.items(), key=lambda item: (item[1], -item[0])
            )
            log.print_fold(
                index=index,
                n_folds=n_folds,
                n_train=len(train),
                n_test=len(test),
                best=best,
                score=best_score,
                elapsed=time.perf_counter() - start,
            )

        sizes = sorted(
            set.intersection(*(set(fold) for fold in fold_scores)), reverse=True
        )
        scores = np.array([[fold[size] for size in sizes] for fold in fold_scores])
        mean_scores = scores.mean(axis=0)

        # Ties go to fewer features, as in sklearn's RFECV
        n_best = sizes[::-1][int(np.argmax(mean_scores[::-1]))]

        self.cv_results_ = {
            "n_features": np.array(sizes),
            "mean_test_score": mean_scores,
            "std_test_score": scores.std(axis=0),
            **{f"split{i}_test_score": split for i, split in enumerate(scores)},
        }

        log.print_cv_scores(
            sizes=sizes,
            mean_scores=mean_scores,
            std_scores=scores.std(axis=0),
            selected=n_best,
        )
        monitor = Monitoring(
            scorer=None, history=History(scored=False), log=log, start=start
        )
        for fold_monitor in fold_monitors:
            monitor.fit_time += fold_monitor.fit_time
            monitor.solve_time += fold_monitor.solve_time
            monitor.score_time += fold_monitor.score_time
            monitor.n_feasible_probes += fold_monitor.n_feasible_probes
            monitor.n_infeasible_probes += fold_monitor.n_infeasible_probes

        build_time = self._setup(
            X=X,
            feature_names=feature_names,
            non_numeric_features=non_numeric_features,
            n_select=n_best,
        )
        self._print_header(
            monitor=monitor,
            X=X_rfe,
            n_select=n_best,
            build_time=build_time,
            title=f"Final fit with {n_best} features",
        )
        support, ranking, termination = self._run(
            X=X_rfe,
            y=y,
            feature_names=feature_names,
            n_select=n_best,
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
        # Sizes outside the cross-validated set have no mean score
        mean_by_size = dict(zip(sizes, mean_scores.tolist()))
        self.history_["score"] = np.array(
            [mean_by_size.get(n, np.nan) for n in self.history_["n_features"].tolist()],
            dtype=float,
        )
        self._summarize(monitor=monitor, termination=termination, cpu_start=cpu_start)

        return self
