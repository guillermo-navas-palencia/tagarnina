import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tagarnina.crfe.crfe import CRFE
from tagarnina.crfe.termination import Termination
from tagarnina.optimization.constraints import (
    AllOrNoneConstraint,
    AtLeastKConstraint,
    AtMostKConstraint,
    Constraints,
    FixedConstraint,
    GroupConstraint,
    KnapsackConstraint,
    MaxCorrelationConstraint,
    MinMaxConstraint,
    OnlyEnforceIfConstraint,
    OnlyExcludeIfConstraint
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(200, 6)),
        columns=[f"x{i}" for i in range(6)]
    )
    y = (X["x0"] + X["x1"] + 0.1 * rng.normal(size=200) > 0).astype(int)
    return X, y


def test_selects_requested_number_of_features(data):
    X, y = data
    selector = CRFE(LogisticRegression(), n_features_to_select=3).fit(X, y)

    assert selector.support_.shape == (6,)
    assert selector.support_.sum() == 3
    assert selector.n_features_ == 3
    assert selector.ranking_[selector.support_].tolist() == [1, 1, 1]
    assert selector.summary_["termination"] == Termination.TARGET_REACHED


def test_transform_keeps_selected_columns(data):
    X, y = data
    selector = CRFE(LogisticRegression(), n_features_to_select=3).fit(X, y)

    X_selected = selector.transform(X)

    assert X_selected.shape == (len(X), 3)
    assert np.array_equal(X_selected, X.to_numpy()[:, selector.support_])
    assert np.array_equal(selector.get_support(), selector.support_)


def test_history_tracks_each_elimination(data):
    X, y = data
    selector = CRFE(LogisticRegression(), n_features_to_select=3).fit(X, y)

    n_features = selector.history_["n_features"]
    assert n_features[0] == 6
    assert n_features[-1] == 3
    assert np.all(np.diff(n_features) <= 0)


def test_fixed_feature_is_kept(data):
    X, y = data
    constraints = Constraints(
        constraints=[FixedConstraint(feature="x5")]
    )
    selector = CRFE(
        LogisticRegression(),
        n_features_to_select=2,
        constraints=constraints
    ).fit(X, y)

    assert selector.support_[5]
    assert selector.support_.sum() == 2


def test_at_most_k_group_is_enforced(data):
    X, y = data
    constraints = Constraints(
        constraints=[
            AtMostKConstraint(features=["x2", "x3", "x4", "x5"], max_features=1)
        ]
    )
    selector = CRFE(
        LogisticRegression(),
        n_features_to_select=3,
        constraints=constraints
    ).fit(X, y)

    assert selector.support_[2:].sum() <= 1


def test_infeasible_constraints_raise(data):
    X, y = data
    constraints = Constraints(
        constraints=[
            AtMostKConstraint(features=["x2", "x3"], max_features=0),
            FixedConstraint(feature="x2")
        ]
    )
    selector = CRFE(
        LogisticRegression(),
        n_features_to_select=2,
        constraints=constraints
    )

    with pytest.raises(ValueError, match="infeasible"):
        selector.fit(X, y)


def test_callback_stops_the_fit(data):
    X, y = data
    selector = CRFE(
        LogisticRegression(),
        n_features_to_select=3,
        callback=lambda row: True
    ).fit(X, y)

    assert selector.summary_["termination"] == Termination.STOPPED
    assert selector.n_features_ > 3


@pytest.mark.parametrize(
    "params",
    [
        {"n_features_to_select": 0},
        {"strategy": "unknown"},
    ]
)
def test_invalid_parameters_raise(data, params):
    X, y = data
    selector = CRFE(LogisticRegression(), **params)

    with pytest.raises(ValueError):
        selector.fit(X, y)


def fit_with(X, y, constraints, n_features_to_select=3):
    return CRFE(
        LogisticRegression(),
        n_features_to_select=n_features_to_select,
        constraints=Constraints(constraints=constraints)
    ).fit(X, y)


def test_max_correlation_keeps_correlated_pair_apart(data):
    X, y = data
    X = X.assign(x1=X["x0"] + 0.01 * X["x2"])
    selector = fit_with(
        X,
        y,
        [MaxCorrelationConstraint(max_corr=0.9, features=["x0", "x1"])]
    )

    assert not (selector.support_[0] and selector.support_[1])


def test_at_least_k_group_selects_only_from_group(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [AtLeastKConstraint(features=["x0", "x1", "x2"], min_features=2)],
        n_features_to_select=2
    )

    assert selector.support_[:3].sum() == 2
    assert selector.support_[3:].sum() == 0


def test_group_constraint_bounds_selection(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [GroupConstraint(features=["x2", "x3", "x4"], min_features=1, max_features=1)]
    )

    assert selector.support_[2:5].sum() == 1


def test_min_max_constraint_bounds_total_selection(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [MinMaxConstraint(min_features=2, max_features=3)]
    )

    assert selector.support_.sum() == 3


def test_only_enforce_if_selects_implied_feature(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [
            FixedConstraint(feature="x0"),
            OnlyEnforceIfConstraint(feature1="x0", feature2="x1"),
        ],
        n_features_to_select=2
    )

    assert selector.support_[0]
    assert selector.support_[1]


def test_only_exclude_if_drops_excluded_feature(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [
            FixedConstraint(feature="x0"),
            OnlyExcludeIfConstraint(feature1="x0", feature2="x1"),
        ],
        n_features_to_select=2
    )

    assert selector.support_[0]
    assert not selector.support_[1]


def test_all_or_none_keeps_group_together(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [AllOrNoneConstraint(features=["x2", "x3", "x4"])]
    )

    assert len(set(selector.support_[2:5].tolist())) == 1


def test_knapsack_limits_total_weight(data):
    X, y = data
    selector = fit_with(
        X,
        y,
        [
            KnapsackConstraint(
                features=["x2", "x3", "x4", "x5"],
                weights=[1.0, 1.0, 1.0, 1.0],
                budget=2.0
            )
        ]
    )

    assert selector.support_.sum() == 3
    assert selector.support_[2:].sum() <= 2


def test_infeasible_report_lists_only_conflicting_constraints(data):
    X, y = data
    constraints = [
        FixedConstraint(feature="x0"),
        AtMostKConstraint(features=["x2", "x3"], max_features=0),
        FixedConstraint(feature="x2"),
    ]

    with pytest.raises(ValueError) as error:
        fit_with(X, y, constraints, n_features_to_select=2)

    message = str(error.value)
    assert "x2 + x3 <= 0" in message
    assert "x2 fixed" in message
    assert "x0 fixed" not in message
