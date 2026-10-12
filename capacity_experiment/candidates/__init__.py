"""Published and explicitly heuristic finite-Dicke input candidates."""

from .families import (
    Candidate,
    RealCombChart,
    gaussian_comb,
    initial_candidates,
    mele_long,
    mele_short,
    mixture_from_rows,
)

__all__ = [
    "Candidate", "RealCombChart", "gaussian_comb", "initial_candidates",
    "mele_long", "mele_short", "mixture_from_rows",
]
