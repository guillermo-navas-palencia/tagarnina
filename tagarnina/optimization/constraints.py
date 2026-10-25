from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Constraint:
    def __str__(self) -> str:
        return repr(self)


@dataclass(kw_only=True)
class MaxCorrelationConstraint(Constraint):
    max_corr: float
    features: list[str] | None = None
    absolute: bool = False
    method: str | Callable = "pearson"

    def __post_init__(self):
        name = type(self).__name__

        if self.features is not None and len(self.features) <= 1:
            raise ValueError(
                f"{name}: features must contain at least two names."
            )

        if self.features is not None and len(set(self.features)) != len(self.features):
            raise ValueError(f"{name}: features must be unique.")

        if not isinstance(self.absolute, bool):
            raise TypeError(
                f"{name}: absolute must be a bool, "
                f"got {type(self.absolute).__name__}."
            )

        if self.method not in ("pearson", "spearman", "kendall") and not callable(
            self.method
        ):
            raise ValueError(
                f"{name}: method must be 'pearson', 'spearman', "
                "'kendall', or a callable."
            )

        if self.absolute:
            if not 0.0 <= self.max_corr <= 1.0:
                raise ValueError(
                    f"{name}: max_corr must be between 0 and 1 "
                    "when absolute is True."
                )
        else:
            if not -1.0 <= self.max_corr <= 1.0:
                raise ValueError(
                    f"{name}: max_corr must be between -1 and 1 "
                    "when absolute is False."
                )


def _validate_bounds(constraint: GroupConstraint | MinMaxConstraint) -> None:
    name = type(constraint).__name__
    if constraint.min_features < 0 or constraint.max_features < 0:
        raise ValueError(
            f"{name}: min_features and max_features must be non-negative."
        )

    if constraint.min_features > constraint.max_features:
        raise ValueError(
            f"{name}: min_features must be less than or equal to max_features."
        )


@dataclass(kw_only=True)
class GroupConstraint(Constraint):
    features: list[str]
    min_features: int
    max_features: int

    def __post_init__(self):
        if not self.features:
            raise ValueError(
                f"{type(self).__name__}: features must contain at least one name."
            )

        if len(set(self.features)) != len(self.features):
            raise ValueError(f"{type(self).__name__}: features must be unique.")

        _validate_bounds(self)

        n_features = len(self.features)
        if n_features < self.min_features:
            raise ValueError(
                f"{type(self).__name__}: min_features cannot exceed the number "
                f"of features ({n_features})."
            )

        if self.max_features > n_features:
            self.max_features = n_features


@dataclass(kw_only=True)
class AtMostKConstraint(GroupConstraint):
    min_features: int = field(default=0, init=False)


@dataclass(kw_only=True)
class AtLeastKConstraint(GroupConstraint):
    max_features: int = field(init=False)

    def __post_init__(self):
        self.max_features = len(self.features)
        super().__post_init__()


@dataclass(kw_only=True)
class MinMaxConstraint(Constraint):
    min_features: int
    max_features: int

    def __post_init__(self):
        _validate_bounds(self)


@dataclass(kw_only=True)
class OnlyEnforceIfConstraint(Constraint):
    feature1: str
    feature2: str

    def __post_init__(self):
        if self.feature1 == self.feature2:
            raise ValueError(
                f"{type(self).__name__}: feature1 and feature2 must be different."
            )


@dataclass(kw_only=True)
class OnlyExcludeIfConstraint(Constraint):
    feature1: str
    feature2: str

    def __post_init__(self):
        if self.feature1 == self.feature2:
            raise ValueError(
                f"{type(self).__name__}: feature1 and feature2 must be different."
            )


@dataclass(kw_only=True)
class FixedConstraint(Constraint):
    feature: str


@dataclass(kw_only=True)
class AllOrNoneConstraint(Constraint):
    features: list[str]

    def __post_init__(self):
        if not self.features:
            raise ValueError(
                f"{type(self).__name__}: features must contain at least one name."
            )


@dataclass(kw_only=True)
class KnapsackConstraint(Constraint):
    """The total weight of the selected features cannot exceed the budget."""

    features: list[str]
    weights: list[float]
    budget: float

    def __post_init__(self):
        name = type(self).__name__

        if not self.features:
            raise ValueError(f"{name}: features must contain at least one name.")

        if len(set(self.features)) != len(self.features):
            raise ValueError(f"{name}: features must be unique.")

        if len(self.features) != len(self.weights):
            raise ValueError(f"{name}: features and weights lengths differ.")

        if not all(math.isfinite(w) and w >= 0 for w in self.weights):
            raise ValueError(f"{name}: weights must be finite and non-negative.")

        if not (math.isfinite(self.budget) and self.budget > 0):
            raise ValueError(f"{name}: budget must be finite and positive.")


@dataclass(kw_only=True)
class Constraints:
    constraints: list[Constraint]

    def __post_init__(self):
        n_min_max = sum(isinstance(c, MinMaxConstraint) for c in self.constraints)
        if n_min_max > 1:
            raise ValueError(
                f"{type(self).__name__}: MinMaxConstraint can appear at most once."
            )

    def validate(
        self,
        *,
        features: list[str],
        non_numeric_features: frozenset[str] = frozenset(),
    ) -> None:
        available_features = set(features)

        for constraint in self.constraints:
            if isinstance(constraint, MinMaxConstraint):
                constraint_features = features
                if constraint.min_features > len(features):
                    raise ValueError(
                        f"{type(constraint).__name__}: min_features cannot exceed "
                        f"the number of features ({len(features)})."
                    )
            elif isinstance(
                constraint,
                (
                    MaxCorrelationConstraint,
                    GroupConstraint,
                    AllOrNoneConstraint,
                    KnapsackConstraint,
                ),
            ):
                constraint_features = (
                    features if constraint.features is None else constraint.features
                )
            elif isinstance(constraint, (OnlyEnforceIfConstraint, OnlyExcludeIfConstraint)):
                constraint_features = [constraint.feature1, constraint.feature2]
            elif isinstance(constraint, FixedConstraint):
                constraint_features = [constraint.feature]
            else:
                raise TypeError(
                    f"Unsupported constraint type: {type(constraint).__name__}"
                )

            unknown_features = [
                feature
                for feature in constraint_features
                if feature not in available_features
            ]
            if unknown_features:
                raise ValueError(
                    f"{type(constraint).__name__} references unknown features: "
                    f"{unknown_features!r}"
                )

            if isinstance(constraint, MaxCorrelationConstraint):
                non_numeric = [
                    feature
                    for feature in constraint_features
                    if feature in non_numeric_features
                ]
                if non_numeric:
                    raise ValueError(
                        f"{type(constraint).__name__} references non-numeric "
                        f"features: {non_numeric!r}"
                    )

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    def __str__(self) -> str:
        return f"CRFEConstraints(constraints={self.constraints!r})"
