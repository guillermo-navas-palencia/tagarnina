"""Solver-style logging of the CRFE fit, through the standard logging module."""

import logging
import math
import sys
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from tagarnina import __version__

logger = logging.getLogger("tagarnina.crfe")

# key, header, width, format
_COLUMNS = (
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

_LABEL_WIDTH = 13

_BANNER = (
    f"tagarnina v{__version__}",
    "Constrained Recursive Feature Elimination",
    "(c) Guillermo Navas-Palencia, 2026",
)


@contextmanager
def verbose_logging(verbose: int) -> Generator[None]:
    """Show the tagarnina log on stdout for the duration of the block."""
    if not verbose:
        yield
        return

    package_logger = logging.getLogger("tagarnina")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    level = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG if verbose >= 2 else logging.INFO)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(level)


def describe_constraints(constraints: Any) -> str:
    counts = Counter(
        type(constraint).__name__.removesuffix("Constraint")
        for constraint in constraints.constraints
    )
    details = ", ".join(f"{name} {count}" for name, count in counts.items())
    return f"{constraints.n_constraints} ({details})"


def _probes(row: dict[str, Any]) -> str | None:
    feasible = row.get("n_feasible_probes")
    if feasible is None or math.isnan(feasible):
        return None
    return f"{int(feasible)}|{int(row['n_infeasible_probes'])}"


def _cell(value: Any, fmt: str, width: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        text = "-"
    elif fmt == "d":
        text = format(int(value), "d")
    else:
        text = format(value, fmt)
    return f"{text:>{width}}"


class FitLog:
    """Banner, iteration table and summary of a fit."""

    def __init__(self, *, scored: bool, solver: bool, probing: bool):
        self.columns = [
            column
            for column in _COLUMNS
            if (scored or column[0] != "score")
            and (solver or column[0] != "solve_time")
            and (probing or column[0] != "probes")
        ]
        self.width = sum(width for _, _, width, _ in self.columns) + 2 * (
            len(self.columns) - 1
        )
        self.enabled = False
        self.iteration = 0

    def _rule(self, char: str) -> None:
        logger.info(char * self.width)

    def _lines(
        self, lines: list[tuple[str, str]], level: int = logging.INFO
    ) -> None:
        for label, value in lines:
            logger.log(level, "%-*s: %s", _LABEL_WIDTH, label, value)

    def _banner(self) -> None:
        self._rule("-")
        for text in _BANNER:
            logger.info(text.center(self.width))
        self._rule("-")

    def start(
        self,
        lines: list[tuple[str, str]],
        details: list[tuple[str, str]] = (),
    ) -> None:
        self.enabled = logger.isEnabledFor(logging.INFO)
        if not self.enabled:
            return

        self._banner()
        self._lines(lines)
        self._lines(details, logging.DEBUG)
        logger.info("")
        self._rule("-")
        logger.info(
            "  ".join(f"{header:>{width}}" for _, header, width, _ in self.columns)
        )
        self._rule("-")

    def row(self, row: dict[str, Any]) -> None:
        if self.enabled:
            values = {"iter": self.iteration, **row, "probes": _probes(row)}
            logger.info(
                "  ".join(
                    _cell(values.get(key), fmt, width)
                    for key, _, width, fmt in self.columns
                )
            )
        self.iteration += 1

    def end(self, lines: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return

        self._rule("-")
        logger.info("")
        self._lines(lines)
        self._rule("=")
