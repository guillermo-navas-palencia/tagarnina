"""Shared base of the constrained recursive feature elimination selectors."""

import numbers
import time
from numbers import Integral, Real
from operator import attrgetter
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    RealNotInt,
    StrOptions,
    validate_parameter_constraints,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from ..optimization.constraints import AtLeastKConstraint, Constraints
from .logger import describe_input, describe_model, describe_settings, describe_solver
from .monitoring import Monitoring
from .solver import CRFESolver
from .termination import Termination


def select_columns(
    X: pd.DataFrame | npt.NDArray, *, columns: npt.NDArray
) -> pd.DataFrame | npt.NDArray:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, columns]
    return X[:, columns]


def column_scorer(
    scorer: Callable, *, X: pd.DataFrame | npt.NDArray, y: npt.NDArray
) -> Callable[[BaseEstimator, npt.NDArray], float]:
    """Score an estimator on the columns given by index, on the rows of X."""

    def score(estimator: BaseEstimator, features: npt.NDArray) -> float:
        return float(scorer(estimator, select_columns(X, columns=features), y))

    return score


def resolve_n_features(
    *, n_features_to_select: int | float | None, n_features: int
) -> int:
    if n_features_to_select is None:
        return n_features // 2
    if isinstance(n_features_to_select, numbers.Integral):
        if not 0 < n_features_to_select <= n_features:
            raise ValueError(
                f"n_features_to_select must be in (0, {n_features}], "
                f"got {n_features_to_select}."
            )
        return int(n_features_to_select)
    if not 0.0 < n_features_to_select <= 1.0:
        raise ValueError(
            "n_features_to_select as a float must be in (0, 1], "
            f"got {n_features_to_select}."
        )
    return max(1, int(n_features_to_select * n_features))


def importance_stats(
    importances: npt.NDArray, *, kept: npt.NDArray
) -> tuple[float, float]:
    """Fraction of the importance kept, and its shortfall with respect to
    keeping the same number of most important features (as RFE would)."""
    total = importances.sum()
    if not total > 0:
        return np.nan, np.nan

    retained = importances[kept].sum() / total
    best = np.sort(importances)[::-1][: kept.sum()].sum() / total
    return float(retained), float(max(best - retained, 0.0))


def get_step(*, step: int | float, n_features: int) -> int:
    if 0.0 < step < 1.0:
        return int(max(1, step * n_features))
    return int(step)


class BaseCRFE(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Elimination loops and attributes shared by `CRFE` and `CRFECV`.

    Subclasses define their parameters, validation specific to them, and
    `fit`.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "step": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
        ],
        "constraints": [Constraints, None],
        "strategy": [StrOptions({"greedy", "probing"})],
        "solver_params": [dict, None],
        "importance_getter": [str, callable],
        "importance_params": [dict, None],
        "importance_offset": [Interval(Real, 0, 0.1, closed="both")],
        "verbose": ["verbose"],
    }

    solver_params_constraints: dict = {
        "num_workers": [Interval(Integral, 1, None, closed="left")],
        "time_limit": [Interval(Real, 0, None, closed="neither")],
    }

    def _fit(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        step: int,
        n_select: int,
        monitor: Monitoring,
        fit_params: dict[str, Any],
    ) -> tuple[npt.NDArray, npt.NDArray, Termination]:
        n_features = X.shape[1]
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        termination = Termination.TARGET_REACHED

        while support.sum() > n_select:
            features = np.flatnonzero(support)
            estimator = clone(self.estimator)
            X_features = select_columns(X, columns=features)

            start = time.perf_counter()
            estimator.fit(X_features, y, **fit_params)
            fit_time = time.perf_counter() - start

            importances = self._get_feature_importances(
                estimator=estimator, X=X_features
            )

            ranks = np.ravel(np.argsort(importances, kind="stable"))
            threshold = min(step, support.sum() - n_select)
            kept = np.ones(len(features), dtype=bool)
            kept[ranks[:threshold]] = False
            retained, gap = importance_stats(importances, kept=kept)

            # Monitoring block
            monitor.fit_time += fit_time

            score = np.nan
            if monitor.scorer is not None:
                start = time.perf_counter()
                score = monitor.scorer(estimator, features)
                monitor.score_time += time.perf_counter() - start

            monitor.record(
                row={
                    "n_features": int(support.sum()),
                    "support": support.copy(),
                    "removed": threshold,
                    "n_feasible_probes": 0,
                    "n_infeasible_probes": 0,
                    "retained_importance": retained,
                    "importance_gap": gap,
                    "fit_time": fit_time,
                    "solve_time": 0.0,
                    "score": score,
                }
            )

            if monitor.stop:
                termination = Termination.STOPPED
                break

            # Update support and ranking
            support[features[ranks[:threshold]]] = False
            ranking[~support] += 1

        return support, ranking, termination

    def _fit_constrained(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        feature_names: list[str],
        step: int,
        n_select: int,
        monitor: Monitoring,
        fit_params: dict[str, Any],
    ) -> tuple[npt.NDArray, npt.NDArray, Termination]:
        n_features = len(feature_names)
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        termination = Termination.TARGET_REACHED

        while True:
            features = np.flatnonzero(support)
            n_remove = min(step, len(features) - n_select)
            if n_remove < 1:
                break

            estimator = clone(self.estimator)
            X_features = select_columns(X, columns=features)

            start = time.perf_counter()
            estimator.fit(X_features, y, **fit_params)
            fit_time = time.perf_counter() - start
            monitor.fit_time += fit_time

            importances = np.zeros(n_features)
            importances[features] = self._get_feature_importances(
                estimator=estimator, X=X_features
            )

            ordered = [
                feature_names[i]
                for i in features[np.argsort(importances[features], kind="stable")]
            ]

            # normalized importances plus a small offset, so that the solver
            # keeps zero-importance features unless constraints force removal
            weights = np.zeros(n_features)
            top = importances[features].max()
            weights[features] = (
                importances[features] / top if top > 0 else 0.0
            ) + self.importance_offset

            start = time.perf_counter()
            result = self.solver_.eliminate(
                weights=weights, ordered=ordered, n_remove=n_remove
            )
            solve_time = time.perf_counter() - start
            monitor.solve_time += solve_time
            monitor.n_feasible_probes += result.n_feasible_probes
            monitor.n_infeasible_probes += result.n_infeasible_probes

            if result.status == "INFEASIBLE":
                termination = Termination.INFEASIBLE
                break
            if result.status not in ("OPTIMAL", "FEASIBLE"):
                raise RuntimeError(
                    f"{type(self).__name__}: the solver returned {result.status}."
                )

            solution = result.solution
            candidates = result.candidates

            # Monitoring block
            retained, gap = importance_stats(importances, kept=support & solution)

            score = np.nan
            if monitor.scorer is not None:
                start = time.perf_counter()
                score = monitor.scorer(estimator, features)
                monitor.score_time += time.perf_counter() - start

            monitor.record(
                row={
                    "n_features": int(support.sum()),
                    "support": support.copy(),
                    "removed": int(support.sum() - (support & solution).sum()),
                    "n_feasible_probes": result.n_feasible_probes,
                    "n_infeasible_probes": result.n_infeasible_probes,
                    "retained_importance": retained,
                    "importance_gap": gap,
                    "fit_time": fit_time,
                    "solve_time": solve_time,
                    "score": score,
                }
            )
            if monitor.stop:
                termination = Termination.STOPPED
                break

            # Update fixed/excluded features, support and ranking
            removed = np.flatnonzero(support & ~solution)
            for index in removed:
                if feature_names[index] in candidates:
                    continue
                self.solver_.build_exclude_constraint(feature=feature_names[index])

            support = support & solution
            ranking[~support] += 1

        return support, ranking, termination

    def _record_selected(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        support: npt.NDArray,
        monitor: Monitoring,
        fit_params: dict[str, Any],
    ) -> BaseEstimator:
        """Fit a clone on the selected features and record the final row."""
        features = np.flatnonzero(support)
        X_selected = select_columns(X, columns=features)
        estimator = clone(self.estimator)

        start = time.perf_counter()
        estimator.fit(X_selected, y, **fit_params)
        fit_time = time.perf_counter() - start
        monitor.fit_time += fit_time

        score = np.nan
        if monitor.scorer is not None:
            start = time.perf_counter()
            score = monitor.scorer(estimator, features)
            monitor.score_time += time.perf_counter() - start

        monitor.record(
            row={
                "n_features": int(support.sum()),
                "support": support.copy(),
                "removed": np.nan,
                "n_feasible_probes": np.nan,
                "n_infeasible_probes": np.nan,
                "retained_importance": np.nan,
                "importance_gap": np.nan,
                "fit_time": fit_time,
                "solve_time": np.nan,
                "score": score,
            }
        )
        return estimator

    def _set_selected_features(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        support: npt.NDArray,
        ranking: npt.NDArray,
        monitor: Monitoring,
        fit_params: dict[str, Any],
    ) -> None:
        self.estimator_ = self._record_selected(
            X=X, y=y, support=support, monitor=monitor, fit_params=fit_params
        )
        self.history_ = monitor.history.to_dict()
        self.n_features_ = support.sum()
        self.support_ = support
        self.ranking_ = ranking

    def _check_input(
        self, *, X: pd.DataFrame | npt.NDArray, y: pd.Series | npt.NDArray
    ) -> tuple[
        pd.DataFrame | npt.NDArray, npt.NDArray, npt.NDArray, list[str], frozenset
    ]:
        """Validate the data and return the input used for display and feature
        names, the validated X and y, and the non-numeric DataFrame columns."""
        feature_names = None
        non_numeric_features = frozenset()

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            if not all(isinstance(name, str) for name in feature_names):
                raise TypeError("DataFrame column names must all be strings.")

            non_numeric_features = frozenset(
                name
                for name, dtype in X.dtypes.items()
                if not pd.api.types.is_numeric_dtype(dtype)
            )

        X_input = X
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

        X_rfe = X_input if isinstance(X_input, pd.DataFrame) else X
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        return X_rfe, X, y, feature_names, non_numeric_features

    def _setup(
        self,
        *,
        X: npt.NDArray,
        feature_names: list[str],
        non_numeric_features: frozenset,
        n_select: int,
    ) -> float:
        """Set `solver_` for one run (None when unconstrained) and return the
        time spent building the constraints."""
        if self.constraints is None:
            self.solver_ = None
            return 0.0

        self.constraints.validate(
            features=feature_names, non_numeric_features=non_numeric_features
        )

        # Translate the n_select features into an at_least_k constraint
        solver_constraints = Constraints(
            constraints=[
                *self.constraints.constraints,
                AtLeastKConstraint(features=feature_names, min_features=n_select),
            ]
        )

        self.solver_ = CRFESolver(
            features=feature_names,
            constraints=solver_constraints,
            strategy=self.strategy,
            **(self.solver_params or {}),
        )

        X_dense = X.toarray() if issparse(X) else X
        X_df = pd.DataFrame(data=X_dense, columns=feature_names).infer_objects()

        build_start = time.perf_counter()
        self.solver_.build_constraints(X=X_df)
        return time.perf_counter() - build_start

    def _run(
        self,
        *,
        X: pd.DataFrame | npt.NDArray,
        y: npt.NDArray,
        feature_names: list[str],
        n_select: int,
        monitor: Monitoring,
        fit_params: dict[str, Any],
    ) -> tuple[npt.NDArray, npt.NDArray, Termination]:
        """Eliminate features down to n_select, after `_setup`."""
        step = get_step(step=self.step, n_features=X.shape[1])

        if self.constraints is None:
            return self._fit(
                X=X,
                y=y,
                step=step,
                n_select=n_select,
                monitor=monitor,
                fit_params=fit_params,
            )
        return self._fit_constrained(
            X=X,
            y=y,
            feature_names=feature_names,
            step=step,
            n_select=n_select,
            monitor=monitor,
            fit_params=fit_params,
        )

    def _describe_input(
        self, *, X: pd.DataFrame | npt.NDArray, n_select: int
    ) -> list[tuple[str, str]]:
        return [
            *describe_input(estimator=self.estimator, X=X),
            *describe_settings(
                step=self.step,
                n_select=n_select,
                constraints=self.constraints,
                strategy=self.strategy,
                scoring=self.scoring,
            ),
        ]

    def _probe_solver(self) -> CRFESolver:
        """Solver with the run settings and no model, to describe it before a fit."""
        return CRFESolver(
            features=[],
            constraints=Constraints(constraints=[]),
            strategy=self.strategy,
            **(self.solver_params or {}),
        )

    def _print_header(
        self,
        *,
        monitor: Monitoring,
        X: pd.DataFrame | npt.NDArray,
        n_select: int,
        build_time: float,
        title: str | None = None,
    ) -> None:
        details = []
        if self.solver_ is not None:
            details = [
                *describe_solver(solver=self.solver_),
                *describe_model(solver=self.solver_, build_time=build_time),
            ]
        monitor.log.print_header(
            lines=self._describe_input(X=X, n_select=n_select),
            details=details,
            title=title,
        )

    def _summarize(
        self, *, monitor: Monitoring, termination: Termination, cpu_start: float
    ) -> None:
        wall_time = time.perf_counter() - monitor.start
        cpu_time = time.process_time() - cpu_start
        n_blocked = 0 if self.solver_ is None else len(self.solver_.blocked)

        self.summary_ = {
            "termination": termination,
            "wall_time": wall_time,
            "cpu_time": cpu_time,
            "fit_time": monitor.fit_time,
            "solve_time": monitor.solve_time,
            "score_time": monitor.score_time,
            "n_feasible_probes": monitor.n_feasible_probes,
            "n_infeasible_probes": monitor.n_infeasible_probes,
            "n_blocked": n_blocked,
        }

        times = [f"fit {monitor.fit_time:.3f}"]
        if self.solver_ is not None:
            times.append(f"solve {monitor.solve_time:.3f}")
        if monitor.scorer is not None:
            times.append(f"score {monitor.score_time:.3f}")

        lines = [
            ("Termination", termination),
            ("Selected", f"{int(self.n_features_)} of {self.n_features_in_} features"),
        ]

        if self.solver_ is not None and self.strategy == "probing":
            lines.append(
                (
                    "Probes",
                    f"{monitor.n_feasible_probes} feasible, "
                    f"{monitor.n_infeasible_probes} infeasible, {n_blocked} blocked",
                )
            )

        lines.append(
            ("Time", f"wall {wall_time:.3f}s ({', '.join(times)}), cpu {cpu_time:.3f}s")
        )

        monitor.log.print_summary(lines=lines)

    def _get_feature_importances(
        self, *, estimator: BaseEstimator, X: npt.NDArray
    ) -> npt.NDArray:
        getter = self.importance_getter
        params = self.importance_params or {}
        if callable(getter):
            return np.abs(np.asarray(getter(estimator=estimator, X=X, **params)))

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
                    f"{type(self).__name__}: importance_getter='auto' requires "
                    "the estimator to expose coef_ or feature_importances_."
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
            caller_name=type(self).__name__,
        )

        params = self.solver_params or {}
        unknown = sorted(set(params) - set(self.solver_params_constraints))
        if unknown:
            raise ValueError(
                f"{type(self).__name__}: unknown solver_params {unknown}, valid "
                f"keys are {sorted(self.solver_params_constraints)}."
            )
        validate_parameter_constraints(
            self.solver_params_constraints,
            params,
            caller_name=f"{type(self).__name__} solver_params",
        )
