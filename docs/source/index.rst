tagarnina
=========

**tagarnina** is a Python library for feature selection with user-defined constraints. It implements constrained recursive feature elimination (CRFE), where a constraint solver decides which features can be removed at each step, and cross-validated CRFE (CRFECV).

Algorithms
----------

- ``CRFE``: recursive feature elimination with constraints. Without constraints it behaves like scikit-learn's ``RFE``. Strategies: ``"probing"`` (default) and ``"greedy"``.
- ``CRFECV``: ``CRFE`` with cross-validation to choose the number of features.

Constraints
-----------

Defined in ``tagarnina.optimization.constraints``:

- ``MaxCorrelationConstraint``: limits the pairwise correlation between selected features.
- ``GroupConstraint``, ``AtMostKConstraint``, ``AtLeastKConstraint``: bound the number of selected features within a group.
- ``MinMaxConstraint``: bounds the total number of selected features.
- ``OnlyEnforceIfConstraint``: if ``feature1`` is selected, ``feature2`` must be selected.
- ``OnlyExcludeIfConstraint``: if ``feature1`` is selected, ``feature2`` cannot be selected.
- ``FixedConstraint``: always selects a feature.
- ``AllOrNoneConstraint``: selects all or none of a set of features.
- ``KnapsackConstraint``: limits the weighted sum of the selected features.

Installation
------------

``tagarnina`` requires Python 3.12 or newer. From a local clone of the repository::

    git clone https://github.com/guillermo-navas-palencia/tagarnina.git
    cd tagarnina
    pip install .

To install the development tools (tests, linting, type checking)::

    pip install ".[dev]"

To build the documentation and run the examples (adds Sphinx, matplotlib, and SHAP)::

    pip install ".[docs]"

Dependencies
------------

Installing the package installs its required dependencies:

- `NumPy <https://numpy.org/>`_
- `pandas <https://pandas.pydata.org/>`_
- `OR-Tools <https://developers.google.com/optimization>`_ (CP-SAT solver)
- `scikit-learn <https://scikit-learn.org/>`_

Optional extras:

- ``dev``: pytest, pytest-cov, ruff, mypy
- ``docs``: sphinx, pydata-sphinx-theme, numpydoc, sphinx-gallery, matplotlib, shap

Citation
--------

If you use ``tagarnina`` in academic work, please cite it::

    @software{navas_palencia_tagarnina,
      author  = {Navas-Palencia, Guillermo},
      title   = {tagarnina: Python library for feature selection with user-defined constraints},
      year    = {2026},
      url     = {https://github.com/guillermo-navas-palencia/tagarnina},
    }

License
-------

Distributed under the Apache License 2.0. See `LICENSE <https://github.com/guillermo-navas-palencia/tagarnina/blob/main/LICENSE>`_ for details.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   api

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Technical

   technical/algorithm
   technical/constraints
   technical/solver
