"""A separate Hermitian spectral enclosure from polar decomposition and Weyl.

Only algorithm='approx' is used to propose eigenvectors/eigenvalues. Acceptance
depends solely on rigorous residual and Gram-defect norms evaluated with Arb.
The certified acb_mat eig routines are never called here. See derivation.md.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

try:
    import flint
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "capacity_article" / "replay_vendor"))
    import flint

from flint import arb, acb, acb_mat, fmpq


def record(value: arb) -> dict:
    return {
        "ball": value.str(35, more=True),
        "lower_rational": str(value.lower().fmpq()),
        "upper_rational": str(value.upper().fmpq()),
    }


def rational(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def frobenius_upper(matrix: acb_mat) -> arb:
    """A nonnegative exact dyadic upper bound on every contained matrix norm."""
    squared = arb(0)
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            upper = abs(matrix[i, j]).upper()
            if not upper.is_finite():
                raise ValueError("Nonfinite residual entry.")
            squared += upper**2
    return squared.sqrt().upper()


def exact_zero(value: acb) -> bool:
    return value.real.is_exact() and value.imag.is_exact() and value.real.fmpq() == 0 and value.imag.fmpq() == 0


def exact_components(matrix: acb_mat) -> list[list[int]]:
    """Independent disjoint-set implementation; both triangles must be zero."""
    n = matrix.nrows()
    if n != matrix.ncols() or n == 0:
        raise ValueError("A nonempty square matrix is required.")
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i):
            if not (exact_zero(matrix[i, j]) and exact_zero(matrix[j, i])):
                a, b = root(i), root(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    result = {}
    for i in range(n):
        result.setdefault(root(i), []).append(i)
    return sorted(result.values(), key=lambda indices: indices[0])


def entropy_on_interval(value: arb) -> arb:
    """Concave scalar entropy range, using endpoints and its unique maximum."""
    if not value.is_finite():
        raise ValueError("Nonfinite spectral interval.")
    lower = max(Fraction(0), Fraction(str(value.lower().fmpq())))
    upper = min(Fraction(1), Fraction(str(value.upper().fmpq())))
    if lower > upper:
        raise ValueError("Enclosure misses the exact physical spectrum [0,1].")
    log_two = arb(2).log()

    def h(endpoint):
        if endpoint in (0, 1):
            return arb(0)
        x = rational(endpoint)
        return -x * x.log() / log_two

    left, right = h(lower), h(upper)
    result = left.union(right)
    turning = (-arb(1)).exp()
    if not (rational(upper) < turning or rational(lower) > turning):
        result = result.union(1 / (arb(1).exp() * log_two))
    return result


def residual_component_entropy(matrix: acb_mat) -> tuple[arb, dict]:
    """Enclose ordered eigenvalues of an exact physical Hermitian matrix.

    The input ball matrix must contain that matrix. Hermiticity/PSD come from
    the channel derivation, not from a successful numerical check here.
    """
    n = matrix.nrows()
    if n != matrix.ncols() or not n:
        raise ValueError("Nonempty square matrix required.")
    started = time.perf_counter()
    approximate, vectors = matrix.eig(right=True, algorithm="approx")
    # Frozen dyadic approximants: uncertainty in the proposed solver output is
    # discarded deliberately because no assertion relies on its accuracy.
    order = sorted(range(n), key=lambda i: Fraction(str(approximate[i].real.mid().fmpq())))
    diagonal = [approximate[i].real.mid() for i in order]
    q = acb_mat([[vectors[i, j].mid() for j in order] for i in range(n)])
    if any(not q[i, j].is_finite() for i in range(n) for j in range(n)):
        raise ValueError("Nonfinite approximate eigenvectors.")
    if any(not d.is_finite() for d in diagonal):
        raise ValueError("Nonfinite approximate eigenvalues.")
    if any(not q[i, j].real.is_exact() or not q[i, j].imag.is_exact()
           for i in range(n) for j in range(n)):
        raise ValueError("Approximate eigenvector centers are not exact dyadics.")
    qh = q.conjugate().transpose()
    gram = qh * q
    for i in range(n):
        gram[i, i] -= 1
    g = frobenius_upper(gram)
    if not g < 1:
        raise ValueError(f"Gram-defect upper bound is not below one: {g}")
    qd = acb_mat([[q[i, j] * diagonal[j] for j in range(n)] for i in range(n)])
    reconstruction = matrix - qd * qh
    residual = frobenius_upper(reconstruction)
    norm_diagonal = max((abs(d).upper() for d in diagonal), key=lambda x: x.fmpq())
    norm_q = (1 + g).sqrt()
    polar_distance = g / (1 + (1 - g).sqrt())
    delta = (residual + norm_diagonal * (norm_q + 1) * polar_distance).upper()
    if not delta.is_finite() or delta < 0:
        raise ValueError("Invalid Weyl error bound.")
    entropy = arb(0)
    crossing = 0
    for center in diagonal:
        enclosed = center + arb(0, delta)
        if not enclosed > 0:
            crossing += 1
        entropy += entropy_on_interval(enclosed)
    return entropy, {
        "dimension": n,
        "gram_defect_frobenius_upper": str(g.fmpq()),
        "reconstruction_residual_frobenius_upper": str(residual.fmpq()),
        "diagonal_operator_norm": str(norm_diagonal.fmpq()),
        "weyl_radius_upper": str(delta.fmpq()),
        "minimum_ordered_center": str(diagonal[0].fmpq()),
        "maximum_ordered_center": str(diagonal[-1].fmpq()),
        "zero_crossing_eigenvalue_intervals": crossing,
        "entropy_bits": record(entropy),
        "seconds": time.perf_counter() - started,
    }


def residual_entropy(matrix: acb_mat, *, algorithm=None) -> tuple[arb, dict]:
    """Full entropy, optionally accelerated by rigorously exact direct sums."""
    if algorithm is not None:
        raise ValueError("This method has no certified-eigensolver selector.")
    entropy = arb(0)
    components = exact_components(matrix)
    diagnostics = []
    for indices in components:
        block = acb_mat([[matrix[i, j] for j in indices] for i in indices])
        part, diagnostic = residual_component_entropy(block)
        entropy += part
        diagnostic["indices"] = indices
        diagnostics.append(diagnostic)
    return entropy, {
        "dimension": matrix.nrows(),
        "component_dimensions": [len(indices) for indices in components],
        "components": diagnostics,
        "zero_crossing_eigenvalue_intervals": sum(d["zero_crossing_eigenvalue_intervals"] for d in diagnostics),
        "entropy_bits": record(entropy),
        "method": "exact-zero direct sums; approximate eig initializer; rigorous Gram/residual polar bound; ordered Weyl intervals",
    }
