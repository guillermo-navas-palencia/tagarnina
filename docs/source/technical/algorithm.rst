Algorithms
==========

CRFE
----
CRFE (constrained recursive feature elimination) is recursive feature elimination
(RFE) in which a CP-SAT solver decides which features to remove at each iteration,
subject to the constraints. Without constraints, CRFE is the standard RFE: at each
iteration the ``step`` least important features are removed, until the target
number of features is reached.

Elimination loop
~~~~~~~~~~~~~~~~

Let :math:`N` the number of features, :math:`S` be the set of retained features and :math:`k` the number of features
to select. Each iteration:

1. Fits a clone of the estimator on the columns in :math:`S`.
2. Computes the importance :math:`I_i` of each retained feature, from ``coef_`` or
   ``feature_importances_``, or from ``importance_getter`` for each feature :math:`i =1,\ldots,N`. Attribute importances
   are squared, as in RFE.
3. Sets the number of features to remove to :math:`r = \min(\text{step}, |S| - k)`.
4. Solves for the features to keep and removes the others from :math:`S`. The
   strategy below defines how the solver decides.

The loop stops when :math:`|S| = k`, when no removal is feasible, or when the
callback requests a stop. The solver keeps at least :math:`k` features, as an
at-least-k constraint over all features. If ``n_features_to_select`` is None,
:math:`k = 1`.

At each iteration the solver maximizes the importance of the retained features:

.. math::

    \max \sum_{i \in S} w_i x_i,
    \qquad w_i = \frac{I_i}{\max_{j \in S} I_j} + \epsilon

subject to the constraints, at least :math:`k` features retained, and
:math:`x_i = 0` for every feature removed in an earlier iteration or in this one.
The offset :math:`\epsilon` (``importance_offset``, default 1e-3) keeps
zero-importance features unless a constraint requires removing them.

Features removed in an earlier iteration stay removed. ``ranking_`` follows RFE:
selected features have rank 1, and a feature removed earlier has a higher rank.

Strategy
~~~~~~~~

The ``strategy`` parameter is used only with ``constraints``. It decides what
happens when removing the :math:`r` least important retained features is
infeasible. Both strategies start from the same candidates: the :math:`r` least
important retained features that are not blocked.

Greedy
    Adds all the candidates as hard constraints and solves once. If that is
    infeasible, the search stops with ``INFEASIBLE``, even if some candidates
    could be removed on their own.

Probing
    Tests the candidates as one set, using assumptions on one literal per
    feature, so the model is unchanged during the test. If that is infeasible,
    it tests the features one by one, from least to most important, on top of the
    features already accepted in this iteration. A feature that can be removed is
    accepted. A feature that cannot is blocked for the rest of the run, because
    removals only accumulate. The search stops when :math:`r` features are
    accepted or all the features have been tested, and the accepted features
    are then removed. A probe that reaches the time limit is skipped for this
    iteration and is not blocked. If no feature can be removed, the search stops
    with ``INFEASIBLE``.

.. list-table::
    :header-rows: 1

    * -
      - Greedy
      - Probing
    * - Features removed per iteration
      - :math:`r`, or more if a constraint forces further removals
      - Up to :math:`r`
    * - Infeasible candidate set
      - Stops the search
      - Removes the features it can
    * - Solver calls per iteration
      - One
      - One if the candidate set is feasible; otherwise one per tested feature and one final solve
    * - Default
      - No
      - Yes

Greedy can remove more than :math:`r` features when a constraint forces it. For
example, ``OnlyEnforceIfConstraint`` with :math:`x_a \leq x_b` forces the removal of
:math:`a` when :math:`b` is removed. Greedy is cheaper per iteration, but it stops
early when the full step is infeasible. Probing continues with fewer features
removed per iteration instead of stopping.

CRFECV
------

CRFECV chooses the number of features by cross-validation, as in scikit-learn's
RFECV. Each fold runs CRFE on its training rows, with the same constraints and
strategy, so everything above applies. The CV step adds these implications:

- The model is built once per fold and once more for the final run, so the cost is
  about :math:`n_\text{folds} + 1` CRFE runs.
- Data-dependent constraints are computed on the training rows of each fold. For
  example, the correlations used by ``MaxCorrelationConstraint``.
- The full set of features is compared unless the constraints do not allow it, for
  example under a budget. Then it is excluded and CRFECV warns.
- Within each fold, every size reached by the search is scored on the test rows.
  These held-out scores are what CRFECV compares. In CRFE, the score is only a
  training-data record.
- Only sizes reached by all folds are compared. With constraints, a fold can stop
  early when infeasible, so the smallest sizes may not be compared.
- The size with the highest mean held-out score is chosen. Ties go to fewer
  features, as in RFECV.
- The final CRFE run on all the data stops at that size. Its selected features are
  not necessarily the features of any fold. If the chosen size is infeasible on the
  full data, the run stops at the last feasible size, as CRFE does.
