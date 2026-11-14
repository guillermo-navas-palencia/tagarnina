"""
CRFECV on German Credit Data
============================

Choosing the number of features with :class:`~tagarnina.crfe.CRFECV` on the German
credit dataset (1,000 applicants, 20 features). :class:`~tagarnina.crfe.CRFE` runs
inside each cross-validation fold, and every size it reaches is scored on the
held-out rows. The size with the best mean score is selected, and the final run
uses all the data.

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

# %%
# Constraints
# -----------
# An :class:`~tagarnina.optimization.constraints.AllOrNoneConstraint` on the
# encoded columns of each categorical feature keeps the whole feature or removes
# it. A :class:`~tagarnina.optimization.constraints.MaxCorrelationConstraint` on
# limits the pairwise Pearson correlation of the numerical features to 0.5.

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
    features=numerical_features, max_corr=0.5, method="pearson"
)
constraints = Constraints(constraints=[*all_or_none_constraints, max_corr_constraint])

# %%
# The selector
# ------------
# :class:`~tagarnina.crfe.CRFECV` uses 5 stratified folds and ROC AUC as the score.
# Each fold runs CRFE and scores every size on its test rows.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from tagarnina.crfe import CRFECV

selector = CRFECV(
    estimator=RandomForestClassifier(
        max_depth=10, min_samples_leaf=0.2, random_state=0
    ),
    constraints=constraints,
    step=0.1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    scoring="roc_auc",
)

# %%
# Fitting
# -------
# The duration and credit amount features have a correlation of about 0.62, so
# they cannot both be selected. The full set of features therefore violates the
# constraints. Its score exists, because the first round of CRFE is
# unconstrained, but it is not a valid choice. CRFECV excludes it from the
# comparison and warns.

import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    selector.fit(X_transformed, y)
for warning in caught:
    print(f"{warning.category.__name__}: {warning.message}")

# %%
# Cross-validation results
# ------------------------
# :attr:`~tagarnina.crfe.CRFECV.cv_results_` holds the mean and standard deviation
# of the held-out score for each size that all folds reached. The size with the
# best mean is selected, and ties go to fewer features.

cv_results = pd.DataFrame(selector.cv_results_)
print(
    cv_results[["n_features", "mean_test_score", "std_test_score"]]
    .round(3)
    .to_string(index=False)
)
print(f"Selected: {selector.n_features_} features")

# %%
# Selected features
# -----------------
# The final run uses all the data and stops at the selected size. As in the basic
# example, the selection keeps each categorical feature whole and does not hold
# both duration and credit amount.

selected_features = selector.get_feature_names_out()
for feature in categorical_features:
    columns = [c for c in X_transformed.columns if c.startswith(f"{feature}_")]
    n_selected = sum(c in selected_features for c in columns)
    assert n_selected in (0, len(columns))
print(
    "duration and credit_amount:",
    [f for f in ("duration", "credit_amount") if f in selected_features],
)

# %%
# The full set is absent, and the best mean score is at 51 features. The score
# decreases as more features are removed. The plot shows it for each size, with the
# selected size marked.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.errorbar(
    cv_results["n_features"],
    cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
    marker="o",
    capsize=3,
)
ax.axvline(
    selector.n_features_,
    color="gray",
    linestyle="--",
    label=f"selected: {selector.n_features_}",
)
ax.set_xlabel("Number of features")
ax.set_ylabel("ROC AUC (held-out)")
ax.set_title("CRFECV scores by number of features")
ax.legend()
fig.tight_layout()
