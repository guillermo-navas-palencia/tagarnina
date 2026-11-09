Constraints
===========

Define the boolean decision variables :math:`x_i \in \{0, 1\}` for :math:`i=1,\ldots, N`,
where :math:`x_i = 1` if feature :math:`i` is retained and :math:`x_i = 0` if it
is removed. Let :math:`G` denote the set of features of a constraint. A subset of
features is feasible when it satisfies all the constraints below.


MaxCorrelationConstraint
------------------------

Let :math:`\rho_{ij}` be the correlation between features :math:`i` and :math:`j`,
computed on the training data at each fit with the chosen ``method``. If
``absolute`` is True, :math:`|\rho_{ij}|` is used instead. Two features whose
correlation is at least ``max_corr`` cannot be selected together:

.. math::

    x_i + x_j \leq 1 \qquad \text{for all pairs } (i, j) \text{ with } \rho_{ij} \geq \rho_{\max}

Only the features in ``features`` are considered, or all features if
``features`` is None.

GroupConstraint
---------------

Define :math:`L` and :math:`U` as the minimum and maximum number of features
selected from the group :math:`G`, respectively. The group constraint is given by

.. math::

    L \leq \sum_{i \in G} x_i \leq U

If :math:`U` is larger than the size of the group, it is reduced to the size of
the group.

AtMostKConstraint
-----------------

Select at most :math:`K` features from the group :math:`G`. This is a
``GroupConstraint`` with :math:`L = 0`:

.. math::

    \sum_{i \in G} x_i \leq K

AtLeastKConstraint
------------------

Select at least :math:`K` features from the group :math:`G`. This is a
``GroupConstraint`` whose upper bound is the size of the group:

.. math::

    \sum_{i \in G} x_i \geq K

MinMaxConstraint
----------------

Bound the total number of selected features over all features :math:`N`. At most
one ``MinMaxConstraint`` can be used.

.. math::

    L \leq \sum_{i=1}^N x_i \leq U

OnlyEnforceIfConstraint
-----------------------

If feature :math:`a` is selected, then feature :math:`b` must be selected too. In
indicator constraint notation:

.. math::

    x_a = 1 \Rightarrow x_b = 1

CP-SAT handles this indicator constraint natively. For a general MIP solver, the
same condition is written as the linear inequality below, which is equivalent
because the variables are binary:

.. math::

    x_a \leq x_b

OnlyExcludeIfConstraint
-----------------------

If feature :math:`a` is selected, then feature :math:`b` cannot be selected. In
indicator constraint notation:

.. math::

    x_a = 1 \Rightarrow x_b = 0

CP-SAT handles this indicator constraint natively. For a general MIP solver, the
same condition is written as the linear inequality below, which is equivalent
because the variables are binary:

.. math::

    x_a + x_b \leq 1

FixedConstraint
---------------

Feature :math:`i` is always selected:

.. math::

    x_i = 1

AllOrNoneConstraint
-------------------

The features in :math:`G` are either all selected or all removed. Take the first
feature :math:`r` of :math:`G` as the reference. In indicator constraint notation,
for every :math:`j \in G` with :math:`j \neq r`:

.. math::

    x_r = 1 \Rightarrow x_j = 1 \qquad \text{and} \qquad x_r = 0 \Rightarrow x_j = 0

CP-SAT handles these indicator constraints natively. For a general MIP solver,
the same condition is written as the linear equality below, which is equivalent
because the variables are binary:

.. math::

    x_i = x_j \qquad \text{for all } i, j \in G

Application
~~~~~~~~~~~

One-hot encoding. A categorical feature such as ``city`` is encoded as one binary
column per category. Those columns form the group :math:`G`. Selecting only some of
them keeps a partial encoding, which the model cannot interpret as a single
categorical feature. ``AllOrNoneConstraint`` keeps the whole encoded feature or
removes it entirely:

.. code-block:: python

    AllOrNoneConstraint(features=["city_paris", "city_london", "city_berlin"])

KnapsackConstraint
------------------

Let :math:`w_i` be the weight of feature :math:`i` and :math:`B` the budget. The
total weight of the selected features in :math:`G` cannot exceed the budget:

.. math::

    \sum_{i \in G} w_i x_i \leq B \qquad \Longleftrightarrow \qquad \sum_{i \in G} \frac{w_i}{B} x_i \leq 1

The solver rounds each coefficient :math:`w_i / B` to the nearest :math:`10^{-6}`,
so the bound holds up to that rounding.

Application
~~~~~~~~~~~

Paid data sources. Using a variable can have a cost, such as a per-record fee for
an external bureau score or a fraud list service, and the scoring run has a limited
budget. Here :math:`w_i` is the cost per record of feature :math:`i`, and
:math:`B` is the budget per record. Features built from internal data have no cost
and are not in :math:`G`.

For example, with two bureau scores costing 0.50 and 0.30, and a fraud list check
costing 0.20, a budget of 0.80 allows any two of the three paid features, but not
all three (total 1.00):

.. code-block:: python

    KnapsackConstraint(
        features=["bureau_score_a", "bureau_score_b", "fraud_list"],
        weights=[0.50, 0.30, 0.20],
        budget=0.80,
    )

To limit the total cost of every feature, include all of them in ``features`` and
give the free features a weight of 0.
