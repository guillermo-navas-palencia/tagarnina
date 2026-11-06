import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tagarnina.crfe.crfecv import CRFECV
from tagarnina.optimization.constraints import Constraints, FixedConstraint


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"x{i}" for i in range(6)])
    y = (X["x0"] + X["x1"] + 0.1 * rng.normal(size=200) > 0).astype(int)
    return X, y


def expected_n_features(cv_results):
    """Largest mean held-out score, ties go to fewer features."""
    mean = cv_results["mean_test_score"]
    sizes = cv_results["n_features"]
    best = np.flatnonzero(mean == mean.max())
    return int(sizes[best].min())


def test_selects_size_with_best_mean_score(data):
    X, y = data
    selector = CRFECV(LogisticRegression(), min_features_to_select=2, cv=3).fit(X, y)

    cv_results = selector.cv_results_
    assert np.all(np.diff(cv_results["n_features"]) < 0)
    assert cv_results["n_features"][0] == 6
    assert cv_results["n_features"][-1] == 2
    assert all(
        cv_results[f"split{i}_test_score"].shape == cv_results["n_features"].shape
        for i in range(3)
    )
    assert selector.n_features_ == expected_n_features(cv_results)
    assert selector.support_.sum() == selector.n_features_

    history = selector.history_
    mean_by_size = dict(
        zip(cv_results["n_features"].tolist(), cv_results["mean_test_score"].tolist())
    )
    assert np.allclose(
        history["score"], [mean_by_size[n] for n in history["n_features"].tolist()]
    )


def test_fixed_feature_is_kept(data):
    X, y = data
    constraints = Constraints(constraints=[FixedConstraint(feature="x5")])
    selector = CRFECV(
        LogisticRegression(), min_features_to_select=2, cv=3, constraints=constraints
    ).fit(X, y)

    assert selector.support_[5]
    assert selector.n_features_ == expected_n_features(selector.cv_results_)


def test_verbose_reports_folds_before_final_fit(data, capsys):
    X, y = data
    CRFECV(LogisticRegression(), min_features_to_select=2, cv=3).fit(X, y)
    assert capsys.readouterr().out == ""

    CRFECV(LogisticRegression(), min_features_to_select=2, cv=3, verbose=1).fit(X, y)
    lines = capsys.readouterr().out.splitlines()
    fold_lines = [i for i, line in enumerate(lines) if line.startswith("Fold ")]
    final_fit = next(i for i, line in enumerate(lines) if line.startswith("Final fit"))
    assert len(fold_lines) == 3
    assert max(fold_lines) < final_fit
    assert any(line.endswith("<- selected") for line in lines)
    assert "Fold 1/3" not in lines

    CRFECV(LogisticRegression(), min_features_to_select=2, cv=3, verbose=2).fit(X, y)
    lines = capsys.readouterr().out.splitlines()
    assert all(f"Fold {i}/3" in lines for i in (1, 2, 3))


def test_verbose_reports_solver_and_model_details(data, capsys):
    X, y = data
    constraints = Constraints(constraints=[FixedConstraint(feature="x5")])

    CRFECV(
        LogisticRegression(),
        min_features_to_select=2,
        cv=3,
        constraints=constraints,
        verbose=1,
    ).fit(X, y)
    lines = capsys.readouterr().out.splitlines()
    assert not any(line.startswith(("Solver", "Model")) for line in lines)

    CRFECV(
        LogisticRegression(),
        min_features_to_select=2,
        cv=3,
        constraints=constraints,
        verbose=2,
    ).fit(X, y)
    lines = capsys.readouterr().out.splitlines()
    solver = [i for i, line in enumerate(lines) if line.startswith("Solver")]
    model = [i for i, line in enumerate(lines) if line.startswith("Model")]
    first_fold = next(i for i, line in enumerate(lines) if line.startswith("Fold 1/3"))
    final_fit = next(i for i, line in enumerate(lines) if line.startswith("Final fit"))
    assert solver[0] < first_fold
    assert len(solver) == 2
    assert len(model) == 4
    assert first_fold < model[0] < model[1] < model[2] < model[3] < final_fit


def test_accepts_numpy_input(data):
    X, y = data
    selector = CRFECV(LogisticRegression(), min_features_to_select=2, cv=3).fit(
        X.to_numpy(), y.to_numpy()
    )

    assert selector.support_.shape == (6,)
    assert selector.n_features_ == expected_n_features(selector.cv_results_)


def test_rejects_invalid_min_features_to_select(data):
    X, y = data
    with pytest.raises(ValueError):
        CRFECV(LogisticRegression(), min_features_to_select=0, cv=3).fit(X, y)


def test_ties_select_fewest_features(data):
    X, y = data
    selector = CRFECV(
        LogisticRegression(),
        min_features_to_select=2,
        cv=3,
        scoring=lambda estimator, X, y: 0.0,
    ).fit(X, y)

    assert selector.n_features_ == 2


def test_accepts_explicit_splits(data):
    X, y = data
    splits = [
        (np.arange(100, 200), np.arange(0, 100)),
        (np.arange(0, 100), np.arange(100, 200)),
    ]
    selector = CRFECV(LogisticRegression(), min_features_to_select=2, cv=splits).fit(
        X, y
    )

    assert "split1_test_score" in selector.cv_results_
    assert "split2_test_score" not in selector.cv_results_


def test_transform_keeps_selected_features(data):
    X, y = data
    selector = CRFECV(LogisticRegression(), min_features_to_select=2, cv=3).fit(X, y)

    assert selector.transform(X).shape == (200, selector.n_features_)
