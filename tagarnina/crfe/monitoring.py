"""State of a CRFE fit: history, log and totals of time and probes."""

import time
from dataclasses import dataclass
from typing import Any, Callable

from .history import History
from .logger import CRFELogger


@dataclass(kw_only=True)
class Monitoring:
    scorer: Callable | None
    history: History
    log: CRFELogger
    start: float
    callback: Callable[[dict[str, Any]], Any] | None = None
    stop: bool = False
    iteration: int = 0
    fit_time: float = 0.0
    solve_time: float = 0.0
    score_time: float = 0.0
    n_feasible_probes: int = 0
    n_infeasible_probes: int = 0

    def record(self, *, row: dict[str, Any]) -> None:
        row = {**row, "elapsed": time.perf_counter() - self.start}
        self.history.append(row)
        self.log.print_iteration(row=row)

        if self.callback is not None and self.callback(
            {**row, "iteration": self.iteration}
        ):
            self.stop = True
        self.iteration += 1
