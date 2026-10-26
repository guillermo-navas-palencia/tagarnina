"""Per-round record of the supports fitted by the elimination."""

from typing import Any

import numpy as np
import numpy.typing as npt

FLOAT_KEYS = (
    "removed",
    "n_feasible_probes",
    "n_infeasible_probes",
    "retained_importance",
    "importance_gap",
    "fit_time",
    "solve_time",
    "elapsed",
)


class History:
    """Rows of the fitted supports, one per round."""

    def __init__(self, *, scored: bool):
        self.scored = scored
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def to_dict(self) -> dict[str, npt.NDArray]:
        keys = [*FLOAT_KEYS, "score"] if self.scored else FLOAT_KEYS

        history = {
            "n_features": np.array([r["n_features"] for r in self.rows], dtype=int),
            "support": np.array([r["support"] for r in self.rows], dtype=bool),
        }
        for key in keys:
            history[key] = np.array([r[key] for r in self.rows], dtype=float)
        return history
