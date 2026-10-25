from typing import Callable

import numpy as np
import pandas as pd

from ortools.sat.python import cp_model

from .constraints import (
    AllOrNoneConstraint,
    Constraints,
    FixedConstraint,
    GroupConstraint,
    KnapsackConstraint,
    MaxCorrelationConstraint,
    MinMaxConstraint,
    OnlyEnforceIfConstraint,
    OnlyExcludeIfConstraint
)

# CP-SAT only accepts integer coefficients, so floats are scaled and rounded
SCALE = 10**6


class CPSolver:
    """CP-SAT model of the feature constraints.

    Algorithms subclass it to add their own variables (see `_build_extra`)
    and their solve logic.
    """

    def __init__(
        self,
        *,
        features: list[str],
        constraints: Constraints,
        num_workers: int = 1,
        time_limit: float = 10.0
    ):
        self.features = features
        self.constraints = constraints

        self.solver = cp_model.CpSolver()
        self.solver.parameters.random_seed = 0
        self.solver.parameters.num_workers = num_workers
        self.solver.parameters.max_time_in_seconds = time_limit

        self.model = cp_model.CpModel()
        self.x = None

    @property
    def n_variables(self) -> int:
        return len(self.model.proto.variables)

    @property
    def n_constraints(self) -> int:
        return len(self.model.proto.constraints)

    def feasibility_check(self) -> None:
        status = self.solver.solve(model=self.model)

        if status == cp_model.INFEASIBLE:
            raise ValueError(
                f"{type(self).__name__}: the constraints are infeasible."
            )

    def build_constraints(
        self,
        *,
        X: pd.DataFrame
    ) -> None:
        self.x = {
            feature: self.model.new_bool_var(name=feature)
            for feature in self.features
        }

        # correlation matrices are shared by constraints using the same method
        corr_cache = {}

        for constraint in self.constraints.constraints:
            if isinstance(constraint, MaxCorrelationConstraint):
                self.build_max_correlation_constraint(
                    X=X, constraint=constraint, corr_cache=corr_cache
                )
            elif isinstance(constraint, GroupConstraint):
                self.build_group_constraint(
                    features=constraint.features, constraint=constraint
                )
            elif isinstance(constraint, MinMaxConstraint):
                self.build_group_constraint(
                    features=self.features, constraint=constraint
                )
            elif isinstance(constraint, OnlyEnforceIfConstraint):
                self.build_only_enforce_if_constraint(constraint=constraint)
            elif isinstance(constraint, OnlyExcludeIfConstraint):
                self.build_only_exclude_if_constraint(constraint=constraint)
            elif isinstance(constraint, FixedConstraint):
                self.build_fixed_constraint(constraint=constraint)
            elif isinstance(constraint, AllOrNoneConstraint):
                self.build_all_or_none_constraint(constraint=constraint)
            elif isinstance(constraint, KnapsackConstraint):
                self.build_knapsack_constraint(constraint=constraint)

        self.feasibility_check()

        self._build_extra()

    def _build_extra(self) -> None:
        """Hook to add algorithm specific variables once the model is built."""

    def build_max_correlation_constraint(
        self,
        *,
        X: pd.DataFrame,
        constraint: MaxCorrelationConstraint,
        corr_cache: dict[str | Callable, pd.DataFrame]
    ) -> None:
        if constraint.method not in corr_cache:
            corr_cache[constraint.method] = X[self.features].corr(
                method=constraint.method, numeric_only=True
            )

        if constraint.features is not None:
            names = constraint.features
        else:
            names = self.features

        corr = corr_cache[constraint.method].loc[names, names].to_numpy()
        if constraint.absolute:
            corr = np.abs(corr)

        rows, cols = np.triu_indices(len(names), k=1)
        conflict = corr[rows, cols] >= constraint.max_corr

        for i, j in zip(rows[conflict], cols[conflict]):
            self.model.add(self.x[names[i]] + self.x[names[j]] <= 1)

    def build_group_constraint(
        self,
        *,
        features: list[str],
        constraint: GroupConstraint | MinMaxConstraint
    ) -> None:
        group_sum = cp_model.LinearExpr.sum(
            [self.x[feature] for feature in features]
        )
        self.model.add_linear_expression_in_domain(
            linear_expr=group_sum,
            domain=cp_model.Domain(
                constraint.min_features,
                constraint.max_features
            )
        )

    def build_only_enforce_if_constraint(
        self,
        *,
        constraint: OnlyEnforceIfConstraint
    ) -> None:
        self.model.add_implication(
            self.x[constraint.feature1], self.x[constraint.feature2]
        )

    def build_only_exclude_if_constraint(
        self,
        *,
        constraint: OnlyExcludeIfConstraint
    ) -> None:
        self.model.add_implication(
            self.x[constraint.feature1], self.x[constraint.feature2].Not()
        )

    def build_fixed_constraint(
        self,
        *,
        constraint: FixedConstraint
    ) -> None:
        self.model.add(self.x[constraint.feature] == 1)

    def build_all_or_none_constraint(
        self,
        *,
        constraint: AllOrNoneConstraint
    ) -> None:
        features = constraint.features
        
        for feature in features[1:]:
            self.model.add(self.x[feature] == self.x[features[0]])

    def build_knapsack_constraint(
        self,
        *,
        constraint: KnapsackConstraint
    ) -> None:
        # weights are relative to the budget, so that the integer scale does not
        # depend on their units
        coefficients = [
            round(weight / constraint.budget * SCALE)
            for weight in constraint.weights
        ]

        self.model.add(
            cp_model.LinearExpr.weighted_sum(
                [self.x[feature] for feature in constraint.features],
                coefficients
            ) <= SCALE
        )
