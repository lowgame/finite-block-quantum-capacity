"""Low-dimensional search families for symmetric depolarizing inputs.

Convention: C[a, k] is the amplitude of |a>_R |D_k^M>_A in a
purification. C has Frobenius norm one. The system density matrix is
C.T @ C.conj(), not C.conj().T @ C, for this amplitude convention.

Published bosonic coefficients are faithfully transcribed. Their finite-M
Dicke lifts and every Gaussian initializer are candidates, not certificates.
No coefficients are attributed to arXiv:2609.39747 beyond what it publishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


CHOIR_URL = "https://arxiv.org/html/2609.39747v2"
MELE_URL = "https://arxiv.org/html/2607.27449v1"

# arXiv:2607.27449v1, Eqs. (10)-(11).
MELE_SHORT_A = (5, -4, 2)
MELE_SHORT_B = (6, -5, 2)
MELE_SHORT_Q = 1 / 3

# arXiv:2607.27449v1, Appendix B, Eqs. (31)-(33).
MELE_LONG_A = (313, -717, 1627, -3027, 4551, -5436, 4983, -3270, 1324, -220)
MELE_LONG_B = (361, -895, 1938, -3422, 4866, -5464, 4646, -2754, 945, -106)
MELE_LONG_Q = 67 / 182


def _as_int(value: int, name: str) -> int:
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def comb_supports(
    M: int, cutoff: int, modulus: int, residues: Sequence[int]
) -> tuple[np.ndarray, ...]:
    M = _as_int(M, "M")
    cutoff = _as_int(cutoff, "cutoff")
    modulus = _as_int(modulus, "modulus")
    residues = tuple(int(a) for a in residues)
    if modulus < 1 or cutoff > M:
        raise ValueError("Require modulus >= 1 and cutoff <= M")
    if len(set(residues)) != len(residues) or not residues:
        raise ValueError("Residues must be nonempty and distinct")
    if any(a < 0 or a >= modulus or a > cutoff for a in residues):
        raise ValueError("Every residue must lie within the modulus and cutoff")
    return tuple(np.arange(a, cutoff + 1, modulus, dtype=int) for a in residues)


def mixture_from_rows(
    M: int,
    supports: Sequence[Sequence[int]],
    coefficients: Sequence[Sequence[complex]],
    weights: Sequence[float],
) -> np.ndarray:
    """Normalize each orthogonal codeword, then apply its mixture weight.

    This normalization is appropriate for separately specified codewords.
    Do NOT use it to load a general preexisting purification payload: such a
    payload must instead be normalized by its total Frobenius norm.
    """
    M = _as_int(M, "M")
    weights = np.asarray(weights, dtype=float)
    if len(supports) != len(coefficients) or len(supports) != len(weights):
        raise ValueError("Supports, coefficient rows and weights must align")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be finite and nonnegative with positive sum")
    weights = weights / weights.sum()
    C = np.zeros((len(supports), M + 1), dtype=np.complex128)
    occupied: set[int] = set()
    for a, (support, values) in enumerate(zip(supports, coefficients)):
        support = np.asarray(support, dtype=int)
        values = np.asarray(values, dtype=np.complex128)
        if len(support) != len(values) or len(support) == 0:
            raise ValueError("Each nonempty support needs one coefficient per site")
        if np.any(support < 0) or np.any(support > M):
            raise ValueError("Support lies outside the finite Dicke space")
        if len(set(support.tolist())) != len(support) or occupied.intersection(support):
            raise ValueError("This family requires disjoint, duplicate-free supports")
        occupied.update(support.tolist())
        norm = np.linalg.norm(values)
        if not np.isfinite(values).all() or norm <= 0:
            raise ValueError("Codeword coefficients must be finite and nonzero")
        C[a, support] = np.sqrt(weights[a]) * values / norm
    return C


def gaussian_comb(
    M: int,
    cutoff: int,
    *,
    modulus: int = 3,
    residues: Sequence[int] = (0, 1),
    means: Sequence[float],
    sigmas: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """A heuristic Gaussian-envelope comb, not the unpublished Choir witness.

    Amplitudes are proportional to exp(-(k-mu)^2/(4 sigma^2)), so squared
    amplitudes have an untruncated Gaussian envelope of width sigma.
    """
    supports = comb_supports(M, cutoff, modulus, residues)
    if len(means) != len(supports) or len(sigmas) != len(supports):
        raise ValueError("One mean and width are required per residue")
    rows = []
    for support, mu, sigma in zip(supports, means, sigmas):
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian means must be finite and widths positive")
        exponent = -((support - mu) / sigma) ** 2 / 4
        # Subtracting the maximum avoids all-underflow for narrow envelopes.
        rows.append(np.exp(exponent - exponent.max()))
    return mixture_from_rows(M, supports, rows, weights)


def mele_short(M: int, *, modulus: int = 4) -> np.ndarray:
    """Eq. (10)-(11), lifted to Dicke basis; modulus != 4 is our heuristic."""
    supports = (np.arange(3) * modulus, np.arange(3) * modulus + 1)
    return mixture_from_rows(
        M, supports, (MELE_SHORT_A, MELE_SHORT_B), (MELE_SHORT_Q, 1 - MELE_SHORT_Q)
    )


def mele_long(M: int, *, modulus: int = 4) -> np.ndarray:
    """Appendix B coefficients, lifted; modulus != 4 is explicitly rescaled."""
    supports = (np.arange(10) * modulus, np.arange(10) * modulus + 1)
    return mixture_from_rows(
        M, supports, (MELE_LONG_A, MELE_LONG_B), (MELE_LONG_Q, 1 - MELE_LONG_Q)
    )


@dataclass
class Candidate:
    name: str
    purification: np.ndarray
    family: str
    provenance: str
    source_url: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        C = self.purification
        populations = np.abs(C) ** 2
        nonzero = np.where(np.any(populations > 0, axis=0))[0]
        return {
            "name": self.name,
            "family": self.family,
            "M": C.shape[1] - 1,
            "rank_bound": C.shape[0],
            "cutoff": int(nonzero[-1]),
            "mean_excitation": float(populations.sum(axis=0) @ np.arange(C.shape[1])),
            "weights": populations.sum(axis=1).tolist(),
            "provenance": self.provenance,
            "source_url": self.source_url,
            "parameters": self.parameters,
            "status": "candidate only; no finite-channel positivity or certification asserted",
        }


def initial_candidates(
    M: int,
    *,
    cutoffs: Iterable[int] = (9, 15, 21),
    include_rank3: bool = False,
) -> list[Candidate]:
    """Small deterministic bank; all handchosen grid points are labeled as ours.

    18 mod-3 Gaussian seeds per requested cutoff, 2--4 published-shape seeds,
    and optionally two rank-three controls per eligible cutoff.
    """
    M = _as_int(M, "M")
    cutoffs = sorted(set(_as_int(K, "cutoff") for K in cutoffs if 1 <= K <= M))
    bank: list[Candidate] = []
    for modulus in (4, 3):
        for tag, size, build in (("short", 3, mele_short), ("long", 10, mele_long)):
            cutoff = modulus * (size - 1) + 1
            if cutoff > M:
                continue
            status = (
                "Published bosonic coefficients exactly transcribed, then lifted to finite Dicke basis."
                if modulus == 4 else
                "Our heuristic compression of published mod-4 coefficient sequence onto mod-3 support."
            )
            bank.append(Candidate(
                f"mele_{tag}_mod{modulus}_M{M}", build(M, modulus=modulus),
                "published_shape", status, MELE_URL,
                {"modulus": modulus, "residues": [0, 1], "cutoff": cutoff},
            ))
    for cutoff in cutoffs:
        for center_fraction in (0.2, 0.4, 0.6):
            for width_fraction in (0.15, 0.3):
                for q in (0.35, 0.5, 0.65):
                    mu = center_fraction * cutoff
                    sigma = max(0.75, width_fraction * cutoff)
                    params = {"cutoff": cutoff, "modulus": 3, "residues": [0, 1],
                              "means": [mu, mu], "sigmas": [sigma, sigma],
                              "weights": [q, 1 - q]}
                    C = gaussian_comb(M, **params)
                    bank.append(Candidate(
                        f"gauss_mod3_K{cutoff}_mu{center_fraction}_sd{width_fraction}_q{q}_M{M}",
                        C, "gaussian_mod3", "Our low-cutoff Gaussian initializer; not published optimized amplitudes.",
                        CHOIR_URL, params,
                    ))
        if include_rank3 and cutoff >= 7:
            for mean_fraction in (0.25, 0.5):
                mu = mean_fraction * cutoff
                sigma = max(1.0, 0.2 * cutoff)
                params = {"cutoff": cutoff, "modulus": 5, "residues": [0, 1, 2],
                          "means": [mu] * 3, "sigmas": [sigma] * 3,
                          "weights": [1 / 3] * 3}
                bank.append(Candidate(
                    f"gauss_mod5_rank3_K{cutoff}_mu{mean_fraction}_M{M}",
                    gaussian_comb(M, **params), "rank3_mod5",
                    "Our initializer; rank-three mod-5 families already occur in the cited paper. Residues here are our choice.",
                    MELE_URL, params,
                ))
    return bank


@dataclass
class RealCombChart:
    """Nonredundant local real-amplitude chart for a disjoint-support seed.

    One nonzero anchor per codeword is fixed to +1 before normalization.
    This covers its local projective chart, not states with zero anchor.
    Mixture weights use rank-1 logits and a configurable interior floor.
    The floor prevents spurious optimization success by collapsing to a pure
    state (whose coherent information is identically zero for any channel).
    """
    M: int
    supports: tuple[np.ndarray, ...]
    anchors: tuple[int, ...]
    weight_floor: float = 0.05

    @classmethod
    def from_seed(
        cls, C: np.ndarray, supports: Sequence[Sequence[int]], *, weight_floor: float = 0.05
    ) -> tuple["RealCombChart", np.ndarray]:
        C = np.asarray(C, dtype=np.complex128)
        supports = tuple(np.asarray(s, dtype=int) for s in supports)
        rank = C.shape[0]
        if len(supports) != rank or not 0 <= weight_floor < 1 / rank:
            raise ValueError("Invalid support count or weight floor")
        if np.max(np.abs(C.imag)) > 1e-12:
            raise ValueError("This chart accepts real coefficients only")
        outside = C.copy()
        for a, support in enumerate(supports):
            outside[a, support] = 0
        if np.max(np.abs(outside)) > 1e-12:
            raise ValueError("Declared supports omit nonzero seed coefficients")
        weights = np.sum(np.abs(C) ** 2, axis=1)
        weights /= weights.sum()
        if np.any(weights <= weight_floor):
            raise ValueError("Every seed weight must exceed weight_floor")
        anchors = tuple(int(np.argmax(np.abs(C[a, s]))) for a, s in enumerate(supports))
        theta: list[float] = []
        for a, (s, anchor) in enumerate(zip(supports, anchors)):
            row = C[a, s].real / C[a, s[anchor]].real
            theta.extend(np.delete(row, anchor).tolist())
        probabilities = (weights - weight_floor) / (1 - rank * weight_floor)
        theta.extend(np.log(probabilities[:-1] / probabilities[-1]).tolist())
        return cls(C.shape[1] - 1, supports, anchors, weight_floor), np.asarray(theta)

    @property
    def n_parameters(self) -> int:
        return sum(len(s) - 1 for s in self.supports) + len(self.supports) - 1

    def decode(self, theta: Sequence[float]) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_parameters,) or not np.isfinite(theta).all():
            raise ValueError("Parameter shape or finiteness mismatch")
        rows = []
        offset = 0
        for s, anchor in zip(self.supports, self.anchors):
            n = len(s) - 1
            row = np.insert(theta[offset:offset + n], anchor, 1.0)
            rows.append(row)
            offset += n
        logits = np.r_[theta[offset:], 0.0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        rank = len(rows)
        weights = self.weight_floor + (1 - rank * self.weight_floor) * probabilities
        return mixture_from_rows(self.M, self.supports, rows, weights)
