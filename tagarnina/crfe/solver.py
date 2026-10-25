import logging

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ortools.sat.python import cp_model

from ..optimization.constraints import Constraints
from ..optimization.solver import SCALE, CPSolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class EliminationResult:
    status: str
    solution: npt.NDArray[np.bool]
    candidates: list[str]
    n_feasible_probes: int
    n_infeasible_probes: int


class CRFESolver(CPSolver):
    """CP-SAT solver of the constrained recursive feature elimination."""

    def __init__(
        self,
        *,
        features: list[str],
        constraints: Constraints,
        strategy: str = "greedy",
        num_workers: int = 1,
        time_limit: float = 10.0
    ):
        super().__init__(
            features=features,
            constraints=constraints,
            num_workers=num_workers,
            time_limit=time_limit
        )
        self.strategy = strategy

        # probing strategy: exclusions are assumptions on drop literals
        self.drop = None
        self.excluded = []
        self.blocked = set()

        self.n_feasible_probes = 0
        self.n_infeasible_probes = 0

    def eliminate(
        self,
        *,
        weights: npt.NDArray,
        ordered: list[str],
        n_remove: int
    ) -> EliminationResult:
        """Exclude up to n_remove features, taken from ordered (worst first),
        and solve. Returns the status, the solution, the excluded features and
        the probe outcomes."""
        n_feasible = self.n_feasible_probes
        n_infeasible = self.n_infeasible_probes
        candidates = [f for f in ordered if f not in self.blocked][:n_remove]

        if self.strategy == "greedy":
            for feature in candidates:
                self.build_exclude_constraint(feature=feature)
            status_name, solution = self.solve(weights=weights)
        else:
            status_name, solution = self.solve(
                weights=weights, trial=candidates
            )
            if status_name == "INFEASIBLE":
                candidates = self.find_removable(ordered=ordered, n_remove=n_remove)
                if candidates:
                    status_name, solution = self.solve(
                        weights=weights, trial=candidates
                    )

            if status_name in ("OPTIMAL", "FEASIBLE"):
                for feature in candidates:
                    self.build_exclude_constraint(feature=feature)

        return EliminationResult(
            status=status_name,
            solution=solution,
            candidates=candidates,
            n_feasible_probes=self.n_feasible_probes - n_feasible,
            n_infeasible_probes=self.n_infeasible_probes - n_infeasible
        )

    def find_removable(
        self,
        *,
        ordered: list[str],
        n_remove: int
    ) -> list[str]:
        """Probe candidates one by one on top of the accepted exclusions."""
        self.model.clear_objective()
        removable = []

        for feature in ordered:
            if len(removable) == n_remove:
                break
            if feature in self.blocked:
                continue

            self.set_assumptions(trial=[*removable, feature])
            status = self.solver.solve(model=self.model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                self.n_feasible_probes += 1
                removable.append(feature)
            elif status == cp_model.INFEASIBLE:
                self.n_infeasible_probes += 1
                # accepted exclusions only grow, so the block is permanent
                self.blocked.add(feature)
            else:
                logger.warning(
                    "probe %s reached the time limit and is skipped.", feature
                )

        return removable

    def set_assumptions(
        self,
        *,
        trial: list[str]
    ) -> None:
        self.model.clear_assumptions()
        for feature in [*self.excluded, *trial]:
            self.model.add_assumption(self.drop[feature])

    def solve(
        self,
        *,
        weights: npt.NDArray,
        trial: list[str] | None = None
    ) -> tuple[str, npt.NDArray[np.bool]]:
        self.model.maximize(
            cp_model.LinearExpr.weighted_sum(
                [self.x[feature] for feature in self.features],
                [round(weight * SCALE) for weight in weights]
            )
        )

        if self.drop is not None:
            self.set_assumptions(trial=trial or [])

        status = self.solver.solve(model=self.model)
        status_name = self.solver.status_name(status=status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = np.array(
                [
                    self.solver.boolean_value(self.x[feature])
                    for feature in self.features
                ],
                dtype=bool
            )
        else:
            solution = np.zeros(len(self.features), dtype=bool)

        return status_name, solution

    def _build_extra(self) -> None:
        if self.strategy != "probing":
            return

        self.drop = {}
        for feature in self.features:
            literal = self.model.new_bool_var(name=f"drop_{feature}")
            self.model.add(self.x[feature] == 0).only_enforce_if(literal)
            self.drop[feature] = literal

    def build_exclude_constraint(
        self,
        *,
        feature: str
    ) -> None:
        if self.drop is None:
            self.model.add(self.x[feature] == 0)
        elif feature not in self.excluded:
            self.excluded.append(feature)
