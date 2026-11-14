"""
Custom importance getter with SHAP values in CRFE
=================================================

:class:`~sklearn.ensemble.HistGradientBoostingClassifier` does not expose ``coef_``
or ``feature_importances_``, so the default ``importance_getter="auto"`` of
:class:`~tagarnina.crfe.CRFE` cannot rank its features. This example passes a
callable instead. The callable returns the mean absolute SHAP value of each
feature, which is a global importance that works for any tree model that SHAP
can explain.

The dataset is the German credit dataset from the basic example, with the same
one-hot encoding and the same all-or-none constraints on the categorical features.

The dataset is public and is available on OpenML (ID 31). Please cite it as:
Dr. Hans Hofmann, Statlog (German Credit Data), UCI Machine Learning Repository,
1994.
"""

# %%
# Load the data and build the constraints
# ---------------------------------------

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
# The default importance getter fails
# -----------------------------------
# With ``importance_getter="auto"`` the first fit raises an error, because the
# estimator exposes neither ``coef_`` nor ``feature_importances_``.

from sklearn.ensemble import HistGradientBoostingClassifier

from tagarnina.crfe import CRFE

estimator = HistGradientBoostingClassifier(random_state=0)

try:
    CRFE(
        estimator=estimator,
        constraints=constraints,
        step=0.1,
        n_features_to_select=20,
        scoring="roc_auc",
    ).fit(X_transformed, y)
except ValueError as error:
    print(error)

# %%
# SHAP importance getter
# ----------------------
# The callable receives the fitted estimator and the current features ``X``.
# It explains the model with :class:`shap.TreeExplainer` and returns the mean
# absolute SHAP value of each feature. CRFE calls it at every elimination step, so
# the importances always refer to the features still in the model.

import numpy as np
import shap


def shap_importances(*, estimator, X):
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    return np.abs(shap_values).mean(axis=0)


# %%
# Feature elimination
# -------------------
# The run is the same as in the basic example, with the callable as
# ``importance_getter``.

crfe = CRFE(
    estimator=estimator,
    importance_getter=shap_importances,
    constraints=constraints,
    step=0.1,
    n_features_to_select=20,
    scoring="roc_auc",
)
crfe.fit(X_transformed, y)

selected_features = crfe.get_feature_names_out()
print(f"Selected {len(selected_features)} features")
print(crfe.summary_["termination"])

# %%
# Explaining the final model
# --------------------------
# :attr:`~tagarnina.crfe.CRFE.estimator_` is fitted on the selected features. The
# bar plot shows their mean absolute SHAP values on the training data.

import matplotlib.pyplot as plt

X_selected = X_transformed[selected_features]
shap_values = shap.TreeExplainer(crfe.estimator_).shap_values(X_selected)

fig, ax = plt.subplots(figsize=(6, 6))
mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=selected_features)
mean_abs.sort_values().plot.barh(ax=ax)
ax.set_xlabel("Mean |SHAP value| (log-odds)")
fig.tight_layout()
