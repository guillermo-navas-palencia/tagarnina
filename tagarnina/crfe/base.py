"""Shared base of the constrained recursive feature elimination selectors."""

import time
from numbers import Integral, Real
from operator import attrgetter
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    RealNotInt,
    StrOptions,
    validate_parameter_constraints
)
from sklearn.utils.validation import check_is_fitted

from ..optimization.constraints import Constraints
from .monitoring import Monitoring
from .termination import Termination


def select_columns(
    X: pd.DataFrame | npt.NDArray,
    *,
    columns: npt.NDArray
) -> pd.DataFrame | npt.NDArray:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, columns]
    return X[:, columns]


def importance_stats(
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
        "scoring": [str, callable, None],
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
        fit_params: dict[str, Any]
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
                score = float(monitor.scorer(estimator, X_features, y))
                monitor.score_time += time.perf_counter() - start

            monitor.record(row={
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
            })

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
        fit_params: dict[str, Any]
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

            start = time.perf_counter()
            result = self.solver_.eliminate(
                weights=weights,
                ordered=ordered,
                n_remove=n_remove
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
                    f"{type(self).__name__}: the solver returned "
                    f"{result.status}."
                )

            solution = result.solution
            candidates = result.candidates

            # Monitoring block
            retained, gap = importance_stats(
                importances, kept=support & solution
            )

            score = np.nan
            if monitor.scorer is not None:
                start = time.perf_counter()
                score = float(monitor.scorer(estimator, X_features, y))
                monitor.score_time += time.perf_counter() - start

            monitor.record(row={
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
            })
            if monitor.stop:
                termination = Termination.STOPPED
                break

            # Update fixed/excluded features, support and ranking
            removed = np.flatnonzero(support & ~solution)
            for index in removed:
                if feature_names[index] in candidates:
                    continue
                self.solver_.build_exclude_constraint(
                    feature=feature_names[index]
                )

            support = support & solution
            ranking[~support] += 1

        return support, ranking, termination

    def _set_selected_features(
        self,
        *,
        X: npt.NDArray | pd.DataFrame,
        y: npt.NDArray,
        support: npt.NDArray,
        ranking: npt.NDArray,
        monitor: Monitoring,
        fit_params: dict[str, Any]
    ) -> None:
        X_selected = select_columns(X, columns=support)
        self.estimator_ = clone(self.estimator)

        start = time.perf_counter()
        self.estimator_.fit(X_selected, y, **fit_params)
        fit_time = time.perf_counter() - start
        monitor.fit_time += fit_time

        score = np.nan
        if monitor.scorer is not None:
            start = time.perf_counter()
            score = float(monitor.scorer(self.estimator_, X_selected, y))
            monitor.score_time += time.perf_counter() - start

        monitor.record(row={
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
        })
        self.history_ = monitor.history.to_dict()
        self.n_features_ = support.sum()
        self.support_ = support
        self.ranking_ = ranking

    def _summarize(
        self,
        *,
        monitor: Monitoring,
        termination: Termination,
        cpu_start: float
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
            (
                "Selected",
                f"{int(self.n_features_)} of {self.n_features_in_} features"
            ),
        ]

        if self.solver_ is not None and self.strategy == "probing":
            lines.append((
                "Probes",
                f"{monitor.n_feasible_probes} feasible, "
                f"{monitor.n_infeasible_probes} infeasible, {n_blocked} blocked"
            ))

        lines.append((
            "Time",
            f"wall {wall_time:.3f}s ({', '.join(times)}), cpu {cpu_time:.3f}s"
        ))

        monitor.log.print_summary(lines=lines)

    def _get_feature_importances(
        self,
        *,
        estimator: BaseEstimator,
        X: npt.NDArray
    ) -> npt.NDArray:
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
            caller_name=type(self).__name__
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
            caller_name=f"{type(self).__name__} solver_params"
        )
