import math

import pytest

from tagarnina.optimization.constraints import (
    AllOrNoneConstraint,
    AtLeastKConstraint,
    AtMostKConstraint,
    Constraint,
    Constraints,
    FixedConstraint,
    GroupConstraint,
    KnapsackConstraint,
    MaxCorrelationConstraint,
    MinMaxConstraint,
    OnlyEnforceIfConstraint,
    OnlyExcludeIfConstraint,
)


@pytest.mark.parametrize(
    "factory, error, match",
    [
        (
            lambda: MaxCorrelationConstraint(max_corr=0.9, features=["x0"]),
            ValueError,
            "at least two",
        ),
        (
            lambda: MaxCorrelationConstraint(max_corr=0.9, features=["x0", "x0"]),
            ValueError,
            "unique",
        ),
        (
            lambda: MaxCorrelationConstraint(max_corr=0.9, absolute="yes"),
            TypeError,
            "absolute must be a bool",
        ),
        (
            lambda: MaxCorrelationConstraint(max_corr=0.9, method="cosine"),
            ValueError,
            "method must be",
        ),
        (
            lambda: MaxCorrelationConstraint(max_corr=1.5),
            ValueError,
            "between -1 and 1",
        ),
        (
            lambda: MaxCorrelationConstraint(max_corr=-0.1, absolute=True),
            ValueError,
            "between 0 and 1",
        ),
        (
            lambda: GroupConstraint(features=[], min_features=0, max_features=1),
            ValueError,
            "at least one",
        ),
        (
            lambda: GroupConstraint(
                features=["x0", "x0"], min_features=0, max_features=1
            ),
            ValueError,
            "unique",
        ),
        (
            lambda: GroupConstraint(
                features=["x0", "x1"], min_features=-1, max_features=1
            ),
            ValueError,
            "non-negative",
        ),
        (
            lambda: GroupConstraint(
                features=["x0", "x1"], min_features=2, max_features=1
            ),
            ValueError,
            "less than or equal",
        ),
        (
            lambda: GroupConstraint(
                features=["x0", "x1"], min_features=3, max_features=3
            ),
            ValueError,
            "cannot exceed",
        ),
        (
            lambda: AtMostKConstraint(features=[], max_features=1),
            ValueError,
            "at least one",
        ),
        (
            lambda: MinMaxConstraint(min_features=-1, max_features=2),
            ValueError,
            "non-negative",
        ),
        (
            lambda: MinMaxConstraint(min_features=3, max_features=2),
            ValueError,
            "less than or equal",
        ),
        (
            lambda: OnlyEnforceIfConstraint(feature1="x0", feature2="x0"),
            ValueError,
            "must be different",
        ),
        (
            lambda: OnlyExcludeIfConstraint(feature1="x0", feature2="x0"),
            ValueError,
            "must be different",
        ),
        (
            lambda: AllOrNoneConstraint(features=[]),
            ValueError,
            "at least one",
        ),
        (
            lambda: KnapsackConstraint(features=[], weights=[], budget=1.0),
            ValueError,
            "at least one",
        ),
        (
            lambda: KnapsackConstraint(
                features=["x0", "x0"], weights=[1.0, 1.0], budget=1.0
            ),
            ValueError,
            "unique",
        ),
        (
            lambda: KnapsackConstraint(
                features=["x0", "x1"], weights=[1.0], budget=1.0
            ),
            ValueError,
            "lengths differ",
        ),
        (
            lambda: KnapsackConstraint(features=["x0"], weights=[-1.0], budget=1.0),
            ValueError,
            "finite and non-negative",
        ),
        (
            lambda: KnapsackConstraint(features=["x0"], weights=[math.nan], budget=1.0),
            ValueError,
            "finite and non-negative",
        ),
        (
            lambda: KnapsackConstraint(features=["x0"], weights=[1.0], budget=0.0),
            ValueError,
            "finite and positive",
        ),
    ],
)
def test_invalid_constraint_raises(factory, error, match):
    with pytest.raises(error, match=match):
        factory()


def test_group_max_features_is_clamped_to_number_of_features():
    constraint = GroupConstraint(features=["x0", "x1"], min_features=0, max_features=5)

    assert constraint.max_features == 2


def test_at_least_k_sets_max_features_to_group_size():
    constraint = AtLeastKConstraint(features=["x0", "x1", "x2"], min_features=1)

    assert constraint.max_features == 3


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (
            MaxCorrelationConstraint(max_corr=0.9),
            "pairwise corr < 0.9 (pearson, all features)",
        ),
        (
            MaxCorrelationConstraint(
                max_corr=0.5,
                features=["x0", "x1"],
                absolute=True,
                method=lambda a, b: 0.0,
            ),
            "pairwise |corr| < 0.5 (custom, 2 features)",
        ),
        (
            GroupConstraint(features=["x0", "x1"], min_features=1, max_features=2),
            "1 <= x0 + x1 <= 2",
        ),
        (AtMostKConstraint(features=["x0", "x1"], max_features=2), "x0 + x1 <= 2"),
        (AtLeastKConstraint(features=["x0", "x1"], min_features=1), "x0 + x1 >= 1"),
        (
            MinMaxConstraint(min_features=2, max_features=3),
            "2 <= sum of all features <= 3",
        ),
        (OnlyEnforceIfConstraint(feature1="x0", feature2="x1"), "x0 => x1"),
        (OnlyExcludeIfConstraint(feature1="x0", feature2="x1"), "x0 => not x1"),
        (FixedConstraint(feature="x0"), "x0 fixed"),
        (AllOrNoneConstraint(features=["x0", "x1"]), "all or none of x0, x1"),
        (
            KnapsackConstraint(features=["x0", "x1"], weights=[1.0, 1.0], budget=2.0),
            "weighted sum of x0, x1 <= 2",
        ),
    ],
)
def test_constraint_string_representation(constraint, expected):
    assert str(constraint) == expected


def test_long_feature_lists_are_truncated_in_string():
    features = [f"x{i}" for i in range(7)]

    assert str(AllOrNoneConstraint(features=features)) == (
        "all or none of x0, ..., x6 (7 features)"
    )
    assert str(AtMostKConstraint(features=features, max_features=2)) == (
        "x0 + ... + x6 (7 features) <= 2"
    )


def test_base_constraint_string_uses_class_name():
    assert str(Constraint()) == "Constraint"


def test_constraints_allow_at_most_one_min_max():
    with pytest.raises(ValueError, match="at most once"):
        Constraints(
            constraints=[
                MinMaxConstraint(min_features=1, max_features=2),
                MinMaxConstraint(min_features=1, max_features=3),
            ]
        )


def test_constraints_counts_and_string():
    constraints = Constraints(
        constraints=[FixedConstraint(feature="x0"), FixedConstraint(feature="x1")]
    )

    assert constraints.n_constraints == 2
    assert str(constraints).startswith("CRFEConstraints(")


@pytest.mark.parametrize(
    "constraint",
    [
        GroupConstraint(features=["x0", "zz"], min_features=0, max_features=1),
        AllOrNoneConstraint(features=["x0", "zz"]),
        KnapsackConstraint(features=["x0", "zz"], weights=[1.0, 1.0], budget=1.0),
        OnlyEnforceIfConstraint(feature1="x0", feature2="zz"),
        OnlyExcludeIfConstraint(feature1="zz", feature2="x0"),
        FixedConstraint(feature="zz"),
        MaxCorrelationConstraint(max_corr=0.9, features=["x0", "zz"]),
    ],
)
def test_validate_rejects_unknown_features(constraint):
    constraints = Constraints(constraints=[constraint])

    with pytest.raises(ValueError, match="unknown features"):
        constraints.validate(features=["x0", "x1"])


def test_validate_rejects_min_features_above_number_of_features():
    constraints = Constraints(
        constraints=[MinMaxConstraint(min_features=3, max_features=3)]
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        constraints.validate(features=["x0", "x1"])


def test_validate_rejects_non_numeric_features_in_max_correlation():
    constraints = Constraints(
        constraints=[MaxCorrelationConstraint(max_corr=0.9, features=["x0", "c0"])]
    )

    with pytest.raises(ValueError, match="non-numeric"):
        constraints.validate(
            features=["x0", "c0"], non_numeric_features=frozenset({"c0"})
        )


def test_validate_rejects_unsupported_constraint_type():
    class CustomConstraint(Constraint):
        pass

    constraints = Constraints(constraints=[CustomConstraint()])

    with pytest.raises(TypeError, match="Unsupported constraint type"):
        constraints.validate(features=["x0", "x1"])


def test_validate_accepts_valid_constraints():
    constraints = Constraints(
        constraints=[
            MaxCorrelationConstraint(max_corr=0.9),
            MinMaxConstraint(min_features=1, max_features=2),
            AtLeastKConstraint(features=["x0", "x1"], min_features=1),
            OnlyEnforceIfConstraint(feature1="x0", feature2="x1"),
            OnlyExcludeIfConstraint(feature1="x0", feature2="x1"),
            FixedConstraint(feature="x0"),
            AllOrNoneConstraint(features=["x0", "x1"]),
            KnapsackConstraint(features=["x0", "x1"], weights=[1.0, 1.0], budget=1.0),
        ]
    )

    constraints.validate(features=["x0", "x1"])
