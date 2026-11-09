Solver
======

The CP-SAT model built from the constraints, the feasibility check run on it, and
the explanation of infeasible constraint sets.

Model
-----

Each feature is a Boolean decision variable, and each constraint is added to a
:class:`~tagarnina.optimization.CPSolver` model as one or more linear constraints
over these variables. ``n_variables`` and ``n_constraints`` report the size of the
model.

The model is built when the selector is fitted. The solver uses a fixed random
seed (0), with the number of workers and the time limit set when it is created.
Algorithms add their own variables through the ``_build_extra`` hook, which runs
after the constraints are added and the feasibility check has passed.

Feasibility check
-----------------

After the constraints are added, the model is solved without an objective. If
the constraints cannot be satisfied together, a ``ValueError`` is raised. Only a
proven infeasibility raises the error. If the solver stops at its time limit
without an answer, no error is raised.

Infeasibility explanation
-------------------------

To find the conflicting constraints, each constraint is built as a separate
group enforced by an assumed literal. CP-SAT returns a subset of these literals
that is sufficient for infeasibility. The subset is then reduced by removing one
constraint at a time while the rest stays infeasible, within the time limit. If
the time limit is reached first, the explanation is marked as not guaranteed
minimal. The error message lists the constraints in this explanation.
