"""
A demo of CRFE on German Credit Data
====================================

Recursive feature elimination with :class:`~tagarnina.crfe.CRFE` on the German
credit dataset (1,000 applicants, 20 features). The one-hot columns of each
categorical feature are kept or removed together, and two numerical features with
a Pearson correlation of at least 0.5 cannot both be selected.

The dataset is public and is available on OpenML (ID 31). Please cite it as:
Dr. Hans Hofmann, Statlog (German Credit Data), UCI Machine Learning Repository,
1994.
"""

# %%
# Load the data
# -------------
# The target is ``good`` or ``bad`` credit risk. It is encoded as 1 for good.

from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=31, as_frame=True, return_X_y=True, parser="auto")
y = (y == "good").astype(int)

categorical_features = X.select_dtypes("category").columns.tolist()
numerical_features = X.select_dtypes("number").columns.tolist()

print(
    f"{len(categorical_features)} categorical and {len(numerical_features)} numerical features"
)

# %%
# One-hot encoding
# ----------------
# Each categorical feature is one-hot encoded. The encoded columns keep the
# original feature name as a prefix, so they can be grouped by name.

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder

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

print(X_transformed.shape)

# %%
# Constraints
# -----------
# An :class:`~tagarnina.optimization.constraints.AllOrNoneConstraint` on the
# encoded columns of each categorical feature keeps the whole feature or removes
# it. A :class:`~tagarnina.optimization.constraints.MaxCorrelationConstraint` on
# the numerical features limits their pairwise correlation.

from tagarnina.optimization.constraints import (
    AllOrNoneConstraint,
    Constraints,
    MaxCorrelationConstraint,
)

all_or_none_constraints = [
    AllOrNoneConstraint(
        features=[c for c in X_transformed.columns if c.startswith(f"{feature}_")]
    )
    for feature in categorical_features
]

max_corr_constraint = MaxCorrelationConstraint(
    features=numerical_features,
    max_corr=0.5,
    method="pearson",
)

constraints = Constraints(
    constraints=[*all_or_none_constraints, max_corr_constraint],
)

# %%
# Feature elimination
# -------------------
# :class:`~tagarnina.crfe.CRFE` refits the estimator at each step and removes
# features until 20 remain. The constraints decide which features can be removed.

from sklearn.ensemble import RandomForestClassifier

from tagarnina.crfe import CRFE

estimator = RandomForestClassifier(max_depth=10, min_samples_leaf=0.2, random_state=0)

crfe = CRFE(
    estimator=estimator,
    constraints=constraints,
    step=0.1,
    n_features_to_select=20,
    scoring="roc_auc",
)
crfe.fit(X_transformed, y)

selected_features = crfe.get_feature_names_out()
print(selected_features)

# %%
# Each categorical feature is either fully selected or fully removed:

for feature in categorical_features:
    columns = [c for c in X_transformed.columns if c.startswith(f"{feature}_")]
    n_selected = sum(c in selected_features for c in columns)
    print(f"{feature}: {n_selected} of {len(columns)} columns selected")

# %%
# The correlation constraint is checked the same way. The check lists the pairs
# of selected numerical features with a correlation of at least 0.5, which
# should be none.

from itertools import combinations

selected_numerical = [f for f in numerical_features if f in selected_features]
corr = X_transformed[selected_numerical].corr(method="pearson")
pairs = [(a, b, corr.loc[a, b]) for a, b in combinations(selected_numerical, 2)]
violations = [(a, b, round(r, 3)) for a, b, r in pairs if r >= 0.5]

print(f"{len(selected_numerical)} numerical features selected, {len(pairs)} pairs")
print(f"Pairs with correlation >= 0.5: {violations}")

# %%
# History
# -------
# :attr:`~tagarnina.crfe.CRFE.history_` has one entry per fitted support, from
# all features down to the 20 selected ones. The fields shown below are:
#
# - ``n_features``: number of features in the support.
# - ``removed``: number of features removed from the support in the next step.
# - ``score``: ROC AUC on the training data.
# - ``importance_gap``: importance lost because of the constraints. It compares
#   the importance kept with the constraints to the importance that standard RFE
#   would keep with the same number of features. It is 0 without constraints.
# - ``n_infeasible_probes``: candidate removals rejected by the constraints.
#
# The last row has no next step, so ``removed`` and ``n_infeasible_probes`` are
# NaN.

import pandas as pd

history = pd.DataFrame(
    {
        key: crfe.history_[key]
        for key in [
            "n_features",
            "removed",
            "score",
            "importance_gap",
            "n_infeasible_probes",
        ]
    }
)
print(history.round(3).to_string(index=False))

# %%
# The score history is plotted below. The score is computed on the training data,
# so it is optimistic and tends to increase with the number of features.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(crfe.history_["n_features"], crfe.history_["score"], marker="o")
ax.set_xlabel("Number of features")
ax.set_ylabel("ROC AUC (training data)")
ax.set_title("CRFE score history")
fig.tight_layout()
