from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Constraint:
    def __str__(self) -> str:
        return type(self).__name__


def format_features(features: list[str], *, sep: str) -> str:
    if len(features) <= 6:
        return sep.join(features)
    return f"{features[0]}{sep}...{sep}{features[-1]} ({len(features)} features)"


@dataclass(kw_only=True)
class MaxCorrelationConstraint(Constraint):
    """Limit the pairwise correlation among the selected features.

    Two features whose correlation is at least `max_corr` cannot be selected
    together.

    Parameters
    ----------
    max_corr : float
        Maximum allowed correlation between two selected features, in [-1, 1],
        or in [0, 1] if `absolute` is True.
    features : list of str, default=None
        Features the constraint applies to. If None, all features are used.
        Must contain at least two unique names.
    absolute : bool, default=False
        If True, the absolute correlation is compared with `max_corr`.
    method : {"pearson", "spearman", "kendall"} or callable, default="pearson"
        Correlation method, as in :meth:`pandas.DataFrame.corr`.
    """

    max_corr: float
    features: list[str] | None = None
    absolute: bool = False
    method: str | Callable = "pearson"

    def __post_init__(self):
        name = type(self).__name__

        if self.features is not None and len(self.features) <= 1:
            raise ValueError(f"{name}: features must contain at least two names.")

        if self.features is not None and len(set(self.features)) != len(self.features):
            raise ValueError(f"{name}: features must be unique.")

        if not isinstance(self.absolute, bool):
            raise TypeError(
                f"{name}: absolute must be a bool, got {type(self.absolute).__name__}."
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
                    f"{name}: max_corr must be between 0 and 1 when absolute is True."
                )
        else:
            if not -1.0 <= self.max_corr <= 1.0:
                raise ValueError(
                    f"{name}: max_corr must be between -1 and 1 when absolute is False."
                )

    def __str__(self) -> str:
        if self.features is None:
            scope = "all features"
        else:
            scope = f"{len(self.features)} features"

        kind = "|corr|" if self.absolute else "corr"
        method = self.method if isinstance(self.method, str) else "custom"
        return f"pairwise {kind} < {self.max_corr:g} ({method}, {scope})"


def _validate_bounds(constraint: GroupConstraint | MinMaxConstraint) -> None:
    name = type(constraint).__name__
    if constraint.min_features < 0 or constraint.max_features < 0:
        raise ValueError(f"{name}: min_features and max_features must be non-negative.")

    if constraint.min_features > constraint.max_features:
        raise ValueError(
            f"{name}: min_features must be less than or equal to max_features."
        )


@dataclass(kw_only=True)
class GroupConstraint(Constraint):
    """Bound the number of selected features within a group.

    The number of selected features in `features` must lie between
    `min_features` and `max_features`. A `max_features` larger than the group
    size is reduced to the group size.

    Parameters
    ----------
    features : list of str
        Names of the features in the group. Must be unique.
    min_features : int
        Minimum number of selected features in the group.
    max_features : int
        Maximum number of selected features in the group.
    """

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

    def __str__(self) -> str:
        return (
            f"{self.min_features} <= {format_features(self.features, sep=' + ')} "
            f"<= {self.max_features}"
        )


@dataclass(kw_only=True)
class AtMostKConstraint(GroupConstraint):
    """Select at most `max_features` features from a group.

    Equivalent to a `GroupConstraint` with `min_features=0`.

    Parameters
    ----------
    features : list of str
        Names of the features in the group. Must be unique.
    max_features : int
        Maximum number of selected features in the group.
    """

    min_features: int = field(default=0, init=False)

    def __str__(self) -> str:
        return f"{format_features(self.features, sep=' + ')} <= {self.max_features}"


@dataclass(kw_only=True)
class AtLeastKConstraint(GroupConstraint):
    """Select at least `min_features` features from a group.

    Equivalent to a `GroupConstraint` with `max_features` equal to the group
    size.

    Parameters
    ----------
    features : list of str
        Names of the features in the group. Must be unique.
    min_features : int
        Minimum number of selected features in the group.
    """

    max_features: int = field(init=False)

    def __post_init__(self):
        self.max_features = len(self.features)
        super().__post_init__()

    def __str__(self) -> str:
        return f"{format_features(self.features, sep=' + ')} >= {self.min_features}"


@dataclass(kw_only=True)
class MinMaxConstraint(Constraint):
    """Bound the total number of selected features.

    At most one `MinMaxConstraint` can be passed to `Constraints`.

    Parameters
    ----------
    min_features : int
        Minimum number of selected features.
    max_features : int
        Maximum number of selected features.
    """

    min_features: int
    max_features: int

    def __post_init__(self):
        _validate_bounds(self)

    def __str__(self) -> str:
        return f"{self.min_features} <= sum of all features <= {self.max_features}"


@dataclass(kw_only=True)
class OnlyEnforceIfConstraint(Constraint):
    """If `feature1` is selected, then `feature2` must be selected too.

    Parameters
    ----------
    feature1 : str
        Feature whose selection implies the selection of `feature2`.
    feature2 : str
        Feature that must be selected when `feature1` is.
    """

    feature1: str
    feature2: str

    def __post_init__(self):
        if self.feature1 == self.feature2:
            raise ValueError(
                f"{type(self).__name__}: feature1 and feature2 must be different."
            )

    def __str__(self) -> str:
        return f"{self.feature1} => {self.feature2}"


@dataclass(kw_only=True)
class OnlyExcludeIfConstraint(Constraint):
    """If `feature1` is selected, then `feature2` must not be selected.

    Parameters
    ----------
    feature1 : str
        Feature whose selection excludes `feature2`.
    feature2 : str
        Feature that cannot be selected when `feature1` is.
    """

    feature1: str
    feature2: str

    def __post_init__(self):
        if self.feature1 == self.feature2:
            raise ValueError(
                f"{type(self).__name__}: feature1 and feature2 must be different."
            )

    def __str__(self) -> str:
        return f"{self.feature1} => not {self.feature2}"


@dataclass(kw_only=True)
class FixedConstraint(Constraint):
    """Always select the feature.

    Parameters
    ----------
    feature : str
        Name of the feature that must be selected.
    """

    feature: str

    def __str__(self) -> str:
        return f"{self.feature} fixed"


@dataclass(kw_only=True)
class AllOrNoneConstraint(Constraint):
    """Select either all of the features or none of them.

    Parameters
    ----------
    features : list of str
        Names of the features that are selected together. Must not be empty.
    """

    features: list[str]

    def __post_init__(self):
        if not self.features:
            raise ValueError(
                f"{type(self).__name__}: features must contain at least one name."
            )

    def __str__(self) -> str:
        return f"all or none of {format_features(self.features, sep=', ')}"


@dataclass(kw_only=True)
class KnapsackConstraint(Constraint):
    """Limit the weighted sum of the selected features.

    The total weight of the selected features cannot exceed `budget`.

    Parameters
    ----------
    features : list of str
        Names of the features. Must be unique.
    weights : list of float
        Non-negative, finite weight of each feature, in the same order as
        `features`.
    budget : float
        Positive, finite budget for the total weight.
    """

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

    def __str__(self) -> str:
        return (
            f"weighted sum of {format_features(self.features, sep=', ')} "
            f"<= {self.budget:g}"
        )


@dataclass(kw_only=True)
class Constraints:
    """Collection of constraints used by `CRFE` and `CRFECV`.

    Parameters
    ----------
    constraints : list of Constraint
        Constraints to enforce during feature elimination.
    """

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
            elif isinstance(
                constraint, (OnlyEnforceIfConstraint, OnlyExcludeIfConstraint)
            ):
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
