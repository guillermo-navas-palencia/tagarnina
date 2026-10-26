"""Printing of the CRFE fit: header, iteration table and summary."""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import ortools

from ..information import banner_lines

if TYPE_CHECKING:
    from ..optimization.solver import CPSolver

ALGORITHM_NAME = "Constrained Recursive Feature Elimination"

# key, header, width, format
COLUMNS = (
    ("iter", "iter", 5, "d"),
    ("n_features", "n_features", 11, "d"),
    ("removed", "removed", 8, "d"),
    ("probes", "probes", 9, "s"),
    ("fit_time", "fit(s)", 8, ".3f"),
    ("solve_time", "solve(s)", 9, ".3f"),
    ("score", "score", 11, ".4e"),
    ("retained_importance", "retained", 9, ".4f"),
    ("importance_gap", "gap", 8, ".4f"),
    ("elapsed", "elapsed(s)", 10, ".3f"),
)

LABEL_WIDTH = 13


def echo(text: str = "") -> None:
    print(text, flush=True)


def describe_input(*, estimator: Any, X: Any) -> list[tuple[str, str]]:
    return [
        ("Estimator", type(estimator).__name__),
        ("Data", f"{X.shape[0]} samples, {X.shape[1]} features"),
    ]


def describe_settings(
    *,
    step: int | float,
    n_select: int,
    constraints: Any,
    strategy: str,
    scoring: str | Callable | None,
) -> list[tuple[str, str]]:
    settings = [f"step={step}", f"n_features_to_select={n_select}"]
    if constraints is not None:
        settings.append(f"strategy={strategy}")
    if scoring is not None:
        name = scoring if isinstance(scoring, str) else "callable"
        settings.append(f"scoring={name}")

    lines = [("Settings", ", ".join(settings))]
    if constraints is not None:
        lines.append(("Constraints", format_constraints(constraints=constraints)))
    return lines


def describe_solver(*, solver: "CPSolver", build_time: float) -> list[tuple[str, str]]:
    params = solver.solver.parameters
    return [
        (
            "Solver",
            f"CP-SAT {ortools.__version__}, workers={params.num_workers}, "
            f"time_limit={params.max_time_in_seconds:g}s, "
            f"seed={params.random_seed}"
        ),
        (
            "Model",
            f"{solver.n_variables} variables, {solver.n_constraints} constraints, "
            f"built in {build_time:.3f}s"
        ),
    ]


def format_constraints(*, constraints: Any) -> str:
    counts = Counter(
        type(constraint).__name__.removesuffix("Constraint")
        for constraint in constraints.constraints
    )
    details = ", ".join(f"{name} {count}" for name, count in counts.items())
    return f"{constraints.n_constraints} ({details})"


def format_probes(*, row: dict[str, Any]) -> str | None:
    feasible = row.get("n_feasible_probes")
    if feasible is None or math.isnan(feasible):
        return None
    return f"{int(feasible)}|{int(row['n_infeasible_probes'])}"


def format_cell(*, value: Any, fmt: str, width: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        text = "-"
    elif fmt == "d":
        text = format(int(value), "d")
    else:
        text = format(value, fmt)
    return f"{text:>{width}}"


class CRFELogger:
    """Header, iteration table and summary of a CRFE fit, printed to stdout.

    `verbose` 0 prints nothing, 1 prints the header, iterations and summary,
    and 2 also prints the solver and model details.
    """

    def __init__(
        self,
        *,
        verbose: int,
        scored: bool,
        solver: bool,
        probing: bool
    ):
        self.verbose = verbose
        self.columns = [
            column
            for column in COLUMNS
            if (scored or column[0] != "score")
            and (solver or column[0] != "solve_time")
            and (probing or column[0] != "probes")
        ]
        self.width = sum(width for _, _, width, _ in self.columns) + 2 * (
            len(self.columns) - 1
        )
        self.iteration = 0

    def print_rule(self, *, char: str) -> None:
        if self.verbose:
            echo(char * self.width)

    def print_lines(
        self,
        *,
        lines: Sequence[tuple[str, str]],
        min_verbose: int = 1
    ) -> None:
        if self.verbose < min_verbose:
            return
        for label, value in lines:
            echo(f"{label:<{LABEL_WIDTH}}: {value}")

    def print_header(
        self,
        *,
        lines: Sequence[tuple[str, str]],
        details: Sequence[tuple[str, str]] = ()
    ) -> None:
        if not self.verbose:
            return

        for text in banner_lines(subtitle=ALGORITHM_NAME, width=self.width):
            echo(text)

        self.print_lines(lines=lines)
        self.print_lines(lines=details, min_verbose=2)
        echo()
        self.print_rule(char="-")
        echo(
            "  ".join(f"{header:>{width}}" for _, header, width, _ in self.columns)
        )
        self.print_rule(char="-")

    def print_iteration(self, *, row: dict[str, Any]) -> None:
        if self.verbose:
            values = {"iter": self.iteration, **row, "probes": format_probes(row=row)}
            echo(
                "  ".join(
                    format_cell(value=values.get(key), fmt=fmt, width=width)
                    for key, _, width, fmt in self.columns
                )
            )
        self.iteration += 1

    def print_summary(self, *, lines: Sequence[tuple[str, str]]) -> None:
        if not self.verbose:
            return

        self.print_rule(char="-")
        echo()
        self.print_lines(lines=lines)
        self.print_rule(char="=")
