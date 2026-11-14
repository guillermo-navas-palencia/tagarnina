"""
Greedy and probing strategies in CRFE
=====================================

:class:`~tagarnina.crfe.CRFE` removes features in steps. At each step it must
choose which of the least important features can be removed together while the
constraints still hold. The ``strategy`` parameter controls how that choice is
made:

- ``"greedy"`` tries to remove all candidates of the step at once. If that is
  infeasible, the search stops.
- ``"probing"`` tests the candidates one by one, from least to most important.
  A candidate that cannot be removed is blocked for the rest of the search, and
  the search continues.

Without constraints both strategies select the same features. The choice matters
when the constraints bind. This example uses the German credit dataset from the
basic example, with an
:class:`~tagarnina.optimization.constraints.AllOrNoneConstraint` on the one-hot
columns of each categorical feature.

The dataset is public and is available on OpenML (ID 31). Please cite it as:
Dr. Hans Hofmann, Statlog (German Credit Data), UCI Machine Learning Repository,
1994.
"""

# %%
# Load the data and build the constraints
# ---------------------------------------
# The one-hot encoding and the all-or-none constraints are the same as in the
# basic example.

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

from tagarnina.optimization.constraints import AllOrNoneConstraint, Constraints

warnings.filterwarnings("ignore")

X, y = fetch_openml(data_id=31, as_frame=True, return_X_y=True, parser="auto")
y = (y == "good").astype(int)
categorical_features = X.select_dtypes("category").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), make_column_selector(dtype_include="category"))
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
X_transformed = pd.DataFrame(
    data=preprocessor.fit_transform(X),
    columns=preprocessor.get_feature_names_out(),
)

constraints = Constraints(
    constraints=[
        AllOrNoneConstraint(
            features=[c for c in X_transformed.columns if c.startswith(f"{feature}_")]
        )
        for feature in categorical_features
    ],
)

# %%
# Run both strategies
# -------------------
# Both runs select 20 features. The summary reports how the search ended, the
# number of infeasible probes, and the number of features blocked from removal.
# Greedy has no probes, so its counts are 0.

from sklearn.ensemble import RandomForestClassifier

from tagarnina.crfe import CRFE


def make_crfe(*, strategy, n_features_to_select, constraints):
    return CRFE(
        estimator=RandomForestClassifier(
            max_depth=10, min_samples_leaf=0.2, random_state=0
        ),
        constraints=constraints,
        step=0.1,
        n_features_to_select=n_features_to_select,
        scoring="roc_auc",
        strategy=strategy,
    )


runs = {}
for strategy in ["greedy", "probing"]:
    crfe = make_crfe(
        strategy=strategy, n_features_to_select=20, constraints=constraints
    ).fit(X_transformed, y)
    runs[strategy] = crfe
    summary = crfe.summary_
    print(
        f"{strategy:8s} termination: {summary['termination']}, "
        f"features: {len(crfe.get_feature_names_out())}, "
        f"infeasible probes: {summary['n_infeasible_probes']}, "
        f"blocked: {summary['n_blocked']}"
    )

# %%
# Greedy stops with more features than requested. Its last step tried to remove
# all candidates at once, and that was infeasible under the constraints. Probing
# tested the same candidates one by one, blocked those it could not remove, and
# reached the target.
#
# The support sizes at each step show where the two searches separate:

for strategy, crfe in runs.items():
    print(strategy, crfe.history_["n_features"].astype(int).tolist())

# %%
# Effect of the target size
# -------------------------
# The gap depends on the target size. The table repeats the run for several target
# sizes. At 15 both strategies reach the target. From 20 upward greedy stops 1 or
# 2 features above the target, while probing reaches each target.

rows = []
for n_features_to_select in [15, 20, 30, 40, 50]:
    for strategy in ["greedy", "probing"]:
        crfe = make_crfe(
            strategy=strategy,
            n_features_to_select=n_features_to_select,
            constraints=constraints,
        ).fit(X_transformed, y)
        rows.append(
            {
                "target": n_features_to_select,
                "strategy": strategy,
                "termination": str(crfe.summary_["termination"]),
                "features": len(crfe.get_feature_names_out()),
                "infeasible_probes": crfe.summary_["n_infeasible_probes"],
            }
        )

results = pd.DataFrame(rows)
print(results.to_string(index=False))

# %%
# The same results as a plot. The dashed line is the target, so points on the line
# reached it.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for strategy, marker in [("greedy", "s"), ("probing", "o")]:
    subset = results[results["strategy"] == strategy]
    ax.plot(subset["target"], subset["features"], marker=marker, label=strategy)
ax.plot(
    results["target"].unique(),
    results["target"].unique(),
    linestyle="--",
    color="grey",
    label="target",
)
ax.set_xlabel("Target number of features")
ax.set_ylabel("Features selected")
ax.legend()
fig.tight_layout()

# %%
# Without constraints the strategies agree
# ----------------------------------------
# With no constraints every candidate can be removed, so greedy and probing make
# the same choices. The same check confirms that the strategy only matters when
# the constraints bind.

unconstrained = {
    strategy: make_crfe(
        strategy=strategy, n_features_to_select=20, constraints=None
    ).fit(X_transformed, y)
    for strategy in ["greedy", "probing"]
}
same = list(unconstrained["greedy"].get_feature_names_out()) == list(
    unconstrained["probing"].get_feature_names_out()
)
print(f"Same selected features without constraints: {same}")
