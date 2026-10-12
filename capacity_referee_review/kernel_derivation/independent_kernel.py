"""Exact-integer fixed-column construction of all depolarizing Schur blocks.

This is new code for the referee review.  It imports no implementation from
capacity_experiment.  In particular it does not use the previous transfer
kernel, its index conventions, its cached coefficient array, or an eigensolver.

For a column with j occupied input bits, use the two-variable polynomial

  (A(t) + E*t*z)**(d-j) * (E + D(t)*z)**j,

where A(t)=a+b*t, D(t)=b+a*t, and a,b,E are exact channel integers with
common denominator L.  The coefficient of z**i is generated below by counting
which of the j input ones stay one.  A sector adds

  (a*b)**q * (1+t)**(2*q),   n=d+2*q.

All polynomial coefficients are integers.  The common channel denominator
is L**n.  Only the normalized Dicke square roots and final state contraction
use Arb arithmetic.  See mathematical_appendix.md for the derivation and
the independent tensor-product oracle in test_independent_kernel.py.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, lcm
from pathlib import Path
import sys
import time

# The host NumPy must be imported before adding the pinned FLINT distribution.
import numpy as np

try:
    import flint
except ImportError:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "capacity_article" / "replay_vendor"))
    import flint

from flint import acb, acb_mat, arb, fmpq, fmpz_poly


def _noise_fraction(value: str | Fraction | int) -> Fraction:
    if isinstance(value, float):
        raise TypeError("Use an exact Fraction or string for the total noise.")
    return Fraction(value)


def _dyadic(value: float) -> fmpq:
    a, b = float(value).as_integer_ratio()
    return fmpq(a, b)


class ExactColumnPolynomials:
    """Integer generating polynomials, without state amplitudes or square roots."""

    def __init__(self, n: int, p_total: str | Fraction):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer.")
        p = _noise_fraction(p_total)
        if not 0 <= p <= Fraction(3, 4):
            raise ValueError("This review covers 0 <= total p <= 3/4.")
        a, b, e = 1 - 2*p/3, 2*p/3, 1 - 4*p/3
        self.n, self.p_total = n, p
        self.denominator = lcm(a.denominator, b.denominator, e.denominator)
        self.a = int(a * self.denominator)
        self.b = int(b * self.denominator)
        self.e = int(e * self.denominator)
        self.channel_denominator = self.denominator ** n
        first = fmpz_poly([self.a, self.b])
        last = fmpz_poly([self.b, self.a])
        self.first_powers = [first**k for k in range(n + 1)]
        self.last_powers = [last**k for k in range(n + 1)]
        self.coherence_powers = [self.e**k for k in range(2*n + 1)]
        self._mixed_powers = {}
        self._sector_factors = {}

    def sector_factor(self, q: int):
        if not 0 <= q <= self.n // 2:
            raise ValueError("Invalid spin sector.")
        if q not in self._sector_factors:
            self._sector_factors[q] = (
                fmpz_poly([1, 1]) ** (2*q) * ((self.a*self.b)**q)
            )
        return self._sector_factors[q]

    def symmetric_column_entry(self, d: int, i: int, j: int):
        """Return [z**i] (A+E*t*z)**(d-j) (E+D*z)**j exactly.

        h is the number of occupied input sites that remain occupied.  The
        normalization sqrt(binomial(d,j)/binomial(d,i)) is applied elsewhere.
        This function also supports i<j, used by the exact tensor oracle.
        """
        if not (0 <= d <= self.n and 0 <= i <= d and 0 <= j <= d):
            raise ValueError("Invalid symmetric-space matrix index.")
        total = fmpz_poly([])
        for h in range(max(0, i+j-d), min(i, j)+1):
            created = i-h
            stayed_zero = d-j-created
            key = (stayed_zero, h)
            if key not in self._mixed_powers:
                self._mixed_powers[key] = (
                    self.first_powers[stayed_zero] * self.last_powers[h]
                )
            ways = comb(d-j, created) * comb(j, h)
            ways *= self.coherence_powers[created+j-h]
            total += self._mixed_powers[key].left_shift(created) * ways
        return total

    def sector_column_entry(self, q: int, i: int, j: int):
        d = self.n-2*q
        return self.sector_factor(q) * self.symmetric_column_entry(d, i, j)


def exact_input_data(coefficients: np.ndarray, n: int, cutoff: int | None = None):
    """Validate the complete input and produce exact dyadic amplitude balls.

    Returning scaled[a][k] = c[a,k]/sqrt(binomial(n,k)) makes the subsequent
    state contraction follow the fixed-column formula directly.
    """
    c = np.asarray(coefficients, dtype=np.complex128)
    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != n+1:
        raise ValueError("A full (reference dimension, n+1) payload is required.")
    if not np.isfinite(c).all():
        raise ValueError("All input coefficients must be finite.")
    kmax = n if cutoff is None else int(cutoff)
    if not 0 <= kmax <= n:
        raise ValueError("The cutoff must lie between zero and n.")
    if np.any(c[:, kmax+1:] != 0):
        raise ValueError("The declared cutoff would remove nonzero coefficients.")
    norm = fmpq(0)
    raw = []
    for row in c:
        converted = []
        for z in row:
            x, y = _dyadic(z.real), _dyadic(z.imag)
            norm += x*x + y*y
            converted.append(acb(arb(x), arb(y)))
        raw.append(converted)
    if norm <= 0:
        raise ValueError("The input has zero norm.")
    roots = [arb(comb(n, k)).sqrt() for k in range(kmax+1)]
    scaled = [[z / roots[k] for k, z in enumerate(row[:kmax+1])] for row in raw]
    return c, kmax, norm, scaled


def iter_weighted_blocks(n: int, p_total: str | Fraction, coefficients: np.ndarray,
                         *, cutoff: int | None = None, precision_bits: int = 512,
                         progress: bool = False):
    """Yield (q, multiplicity-weighted joint block, output block), all sectors.

    Construct AND iterate inside flint.ctx.workprec(precision_bits).  The
    reference precedes the ascending-excitation output index.  No entropy
    routines or original project modules are imported or invoked.
    """
    if flint.ctx.prec != precision_bits:
        raise ValueError("The active Arb precision must match precision_bits.")
    integer_kernel = ExactColumnPolynomials(n, p_total)
    c, kmax, norm, scaled = exact_input_data(coefficients, n, cutoff)
    rank = c.shape[0]

    # Only exact coefficient zeros are omitted.  No magnitude threshold is used.
    products = {}
    for delta in range(kmax+1):
        terms = {}
        for r in range(rank):
            for s in range(rank):
                nonzero = []
                for k in range(delta, kmax+1):
                    if c[r, k] != 0 and c[s, k-delta] != 0:
                        nonzero.append((k, scaled[r][k] * scaled[s][k-delta].conjugate()))
                if nonzero:
                    terms[r, s] = nonzero
        products[delta] = terms

    common_scale = arb(fmpq(1, integer_kernel.channel_denominator)) / arb(norm)
    for q in range(n//2+1):
        begin = time.perf_counter()
        if flint.ctx.prec != precision_bits:
            raise ValueError("Arb precision changed while iterating.")
        d = n-2*q
        size = d+1
        multiplicity = comb(n, q) - (comb(n, q-1) if q else 0)
        joint = acb_mat(rank*size, rank*size)
        for j in range(size):
            for i in range(j, min(d, j+kmax)+1):
                delta = i-j
                if not products[delta]:
                    continue
                polynomial = integer_kernel.sector_column_entry(q, i, j)
                scale = common_scale * multiplicity * arb(fmpq(comb(d, j), comb(d, i))).sqrt()
                for (r, s), terms in products[delta].items():
                    value = acb(0)
                    for k, coefficient_product in terms:
                        integer = polynomial[k]
                        if integer:
                            value += coefficient_product * integer
                    value *= scale
                    joint[r*size+i, s*size+j] = value
                    if i != j:
                        joint[s*size+j, r*size+i] = value.conjugate()
        output = acb_mat(size, size)
        for i in range(size):
            for j in range(size):
                output[i, j] = sum((joint[r*size+i, r*size+j] for r in range(rank)), acb(0))
        if progress:
            import json
            print(json.dumps({"stage": "independent_integer_column_block", "n": n,
                              "q": q, "dimension": size, "seconds": time.perf_counter()-begin}),
                  flush=True)
        yield q, joint, output
