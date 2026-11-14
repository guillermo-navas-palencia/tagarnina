tagarnina
=========

Constrained feature selection.

At each step of recursive feature elimination, the estimator is refit and a
CP-SAT model decides which features can be removed while respecting the
constraints you define, such as group limits, correlation limits, or
implications between features.

Installation
------------

Install from source::

    pip install .

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Technical

   technical/algorithm
   technical/constraints
   technical/solver
