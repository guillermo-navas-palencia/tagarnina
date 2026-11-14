"""
Stopping CRFE early with a callback on superconductor data
==========================================================

:class:`~tagarnina.crfe.CRFE` calls ``callback`` after each fitted support. A
truthy return value stops the elimination and keeps that support. Here the
callback stops the run once the support is in a range of acceptable sizes, the
time budget has been used, and the score has started to drop.

The dataset is the same as in the advanced example: the critical temperature of
21,263 superconductors, available on OpenML (ID 44964) under the CC BY 4.0
licence. Please cite it as: Hamidieh, K., A data-driven statistical model for
predicting the critical temperature of a superconductor, Computational Materials
Science, 154, 346-354 (2018).
"""

# %%
# Load the data
# -------------

from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=44964, as_frame=True, return_X_y=True, parser="auto")

print(X.shape)

# %%
# Constraints
# -----------
# The constraints are the same as in the advanced example: bounds on the range
# and mean statistics, a group bound on the weighted entropy statistics, and a
# link between each weighted mean and its unweighted mean.

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
# Stop rule
# ---------
# The callback returns ``True`` only when all three conditions hold:
#
# * the support has between ``MIN_FEATURES`` and ``MAX_FEATURES`` features,
# * at least ``TIME_CAP`` seconds have elapsed, and
# * the score is more than ``TOLERANCE`` below the best score seen so far.
#
# The window keeps the selection no smaller than the target and no larger than
# the upper bound. The time cap is a budget: the run is not stopped before it
# has been used. The tolerance stops the run when removing features starts to
# cost score. The best score is tracked over all supports, including those
# outside the window.
#
# The score is computed on the training data, so it is optimistic.

MIN_FEATURES = 20
MAX_FEATURES = 40
TIME_CAP = 60  # seconds
TOLERANCE = 0.002


def make_stop_rule():
    best_score = -float("inf")

    def callback(row):
        nonlocal best_score
        best_score = max(best_score, row["score"])
        in_window = MIN_FEATURES <= row["n_features"] <= MAX_FEATURES
        dropped = best_score - row["score"] > TOLERANCE
        return in_window and row["elapsed"] >= TIME_CAP and dropped

    return callback


# %%
# Feature elimination
# -------------------
# A smaller ``step`` removes fewer features at each iteration, so there are more
# supports to check against the stop rule. ``verbose=1`` prints one line per
# fitted support.

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
    step=0.05,
    n_features_to_select=MIN_FEATURES,
    scoring="r2",
    callback=make_stop_rule(),
    verbose=1,
)
crfe.fit(X, y)

n_selected = len(crfe.get_feature_names_out())
print(crfe.summary_["termination"])
print(f"Selected {n_selected} features after {crfe.summary_['wall_time']:.0f} s")

# %%
# The plot shows the score at each support size. The shaded band is the window
# where the callback may stop, and the dashed line marks the selected support.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(
    MIN_FEATURES,
    MAX_FEATURES,
    color="tab:green",
    alpha=0.1,
    label="stop window",
)
ax.plot(crfe.history_["n_features"], crfe.history_["score"], marker="o")
ax.axvline(n_selected, color="tab:red", linestyle="--", label="selected")
ax.set_xlabel("Number of features")
ax.set_ylabel("R2 (training data)")
ax.legend()
fig.tight_layout()
