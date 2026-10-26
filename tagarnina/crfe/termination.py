"""Reasons for which a fit ends."""

from enum import StrEnum


class Termination(StrEnum):
    TARGET_REACHED = "reached the target number of features"
    INFEASIBLE = "no feasible elimination under the constraints"
    STOPPED = "stopped by the callback"
