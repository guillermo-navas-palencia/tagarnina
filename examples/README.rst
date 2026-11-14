Examples
========

Examples of constrained feature selection with tagarnina.

The library currently has two algorithms, :class:`~tagarnina.crfe.CRFE` and
:class:`~tagarnina.crfe.CRFECV`, and the list will evolve. Example titles name the
algorithm they use.

:class:`~tagarnina.crfe.CRFE` without constraints is standard recursive feature
elimination, as in scikit-learn's ``RFE``. The examples show its constrained
extension, where the constraints decide which features can be removed at each
step.
