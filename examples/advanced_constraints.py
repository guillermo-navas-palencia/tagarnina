"""
Advanced constraints in CRFE on superconductor data
===================================================

Recursive feature elimination with :class:`~tagarnina.crfe.CRFE` for a continuous
target: the critical temperature of 21,263 superconductors. The constraints
combine group bounds with implications between features. The verbose output
shows the progress of the elimination.

The dataset is public and is available on OpenML (ID 44964), under the CC BY 4.0
licence. Please cite it as: Hamidieh, K., A data-driven statistical model for
predicting the critical temperature of a superconductor, Computational Materials
Science, 154, 346-354 (2018).
"""

# %%
# Load the data
# -------------
# The target is the critical temperature in kelvin. Each feature is a statistic
# (for example ``mean_``, ``range_`` or ``wtd_entropy_``) of an elemental
# property, named by the statistic followed by the property, such as
# ``range_Density``. The weighted statistics (``wtd_`` prefix) weight each
# element by its fraction in the compound.

from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=44964, as_frame=True, return_X_y=True, parser="auto")

print(X.shape)

# %%
# Constraints
# -----------
# Features are grouped by their statistic, which is the prefix of the name. A
# :class:`~tagarnina.optimization.constraints.AtMostKConstraint` limits the range
# statistics, an :class:`~tagarnina.optimization.constraints.AtLeastKConstraint`
# requires some mean statistics, and a
# :class:`~tagarnina.optimization.constraints.GroupConstraint` sets both bounds on
# the weighted entropy statistics.
#
# An :class:`~tagarnina.optimization.constraints.OnlyEnforceIfConstraint` links
# each weighted mean to the unweighted mean of the same property: the weighted
# mean can be selected only if the unweighted mean is too. This is a modelling
# choice that keeps the two versions of a property together.

from tagarnina.optimization.constraints import (
    AtLeastKConstraint,
    AtMostKConstraint,
    Constraints,
    GroupConstraint,
    OnlyEnforceIfConstraint,
)


def features_with_prefix(prefix):
    return [c for c in X.columns if c.startswith(prefix)]


range_features = features_with_prefix("range_")
mean_features = features_with_prefix("mean_")
wtd_entropy_features = features_with_prefix("wtd_entropy_")
properties = [c.removeprefix("mean_") for c in mean_features]

constraints = Constraints(
    constraints=[
        AtMostKConstraint(features=range_features, max_features=3),
        AtLeastKConstraint(features=mean_features, min_features=2),
        GroupConstraint(
            features=wtd_entropy_features,
            min_features=1,
            max_features=4,
        ),
        *[
            OnlyEnforceIfConstraint(feature1=f"wtd_mean_{p}", feature2=f"mean_{p}")
            for p in properties
        ],
    ],
)

# %%
# Feature elimination
# -------------------
# :class:`~tagarnina.crfe.CRFE` keeps 15 features. ``verbose=1`` prints the
# settings, one line per fitted support, and a summary. ``verbose=2`` also prints
# the details of the solver and the model.

from sklearn.ensemble import RandomForestRegressor

from tagarnina.crfe import CRFE

estimator = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=0,
)

crfe = CRFE(
    estimator=estimator,
    constraints=constraints,
    step=0.1,
    n_features_to_select=15,
    scoring="r2",
    verbose=1,
)
crfe.fit(X, y)

selected_features = crfe.get_feature_names_out()
print(selected_features)

# %%
# Check the constraints on the selected features. The counts below are
# compared with the bounds set above.

selected = set(selected_features)

n_range = sum(c in selected for c in range_features)
n_mean = sum(c in selected for c in mean_features)
n_wtd_entropy = sum(c in selected for c in wtd_entropy_features)
print(f"range statistics: {n_range} selected (at most 3)")
print(f"mean statistics: {n_mean} selected (at least 2)")
print(f"weighted entropy statistics: {n_wtd_entropy} selected (between 1 and 4)")

violations = [
    p for p in properties if f"wtd_mean_{p}" in selected and f"mean_{p}" not in selected
]
print(f"Weighted means without their unweighted mean: {violations}")

# %%
# The run summary records how the elimination ended and how many probes were
# infeasible.

print(crfe.summary_["termination"])
print(f"Features that could not be removed: {crfe.summary_['n_blocked']}")

# %%
# The plots show the score and the importance gap at each support size. The score
# is computed on the training data, so it is optimistic and tends to increase with
# the number of features. The importance gap is the importance lost because of the
# constraints, compared with standard RFE with the same number of features. See
# the basic example for the full history.

import matplotlib.pyplot as plt

n_features = crfe.history_["n_features"]

fig, (ax_score, ax_gap) = plt.subplots(1, 2, figsize=(10, 4))
ax_score.plot(n_features, crfe.history_["score"], marker="o")
ax_score.set_xlabel("Number of features")
ax_score.set_ylabel("R2 (training data)")
ax_score.set_title("Score")

ax_gap.plot(n_features, crfe.history_["importance_gap"], marker="o")
ax_gap.set_xlabel("Number of features")
ax_gap.set_ylabel("Importance gap")
ax_gap.set_title("Importance lost to constraints")
fig.tight_layout()
