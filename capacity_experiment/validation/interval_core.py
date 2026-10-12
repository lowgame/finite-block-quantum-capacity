"""Rigorous ball-arithmetic evaluator for a *fixed* symmetric witness.

This module reconstructs the Appendix E polynomial kernel independently with
Arb real balls, starting from exact rational noise and exact IEEE754 dyadic
coefficients. It uses python-flint's certified acb_mat.eig(multiple=True),
never algorithm='approx', and encloses -x log2 x including eigenvalues near
zero. It includes every spin sector and its multiplicity.

This is a computer-assisted sign bound using FLINT/Arb as trusted numerical
infrastructure. It is not the paper's dependency-free integer certificate,
an optimality proof, or a capacity upper bound.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from math import comb, ceil, log10
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # Load the host NumPy before making vendor/flint visible.

try:
    import flint
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
    import flint

from flint import acb, acb_mat, arb, arb_poly, fmpq


def exact_float(value: float) -> fmpq:
    numerator, denominator = float(value).as_integer_ratio()
    return fmpq(numerator, denominator)


def exact_complex(value: complex) -> acb:
    value = complex(value)
    return acb(arb(exact_float(value.real)), arb(exact_float(value.imag)))


def _fraction(value: str | int | Fraction) -> Fraction:
    if isinstance(value, float):
        raise TypeError("Pass p_total as an explicit rational/decimal string, not a float.")
    return Fraction(value)


def ball_record(value: arb, digits: int = 35) -> dict:
    """Exact rational bounds plus a convenient outward ball rendering."""
    return {
        "ball": value.str(digits, more=True),
        "lower_rational": str(value.lower().fmpq()),
        "upper_rational": str(value.upper().fmpq()),
    }


def entropy_eigenvalue_bound(value: arb) -> arb:
    """Bound h(lambda) using the known physical constraint 0<=lambda<=1."""
    if not value.is_finite() or not value.overlaps(arb("0.5", "0.5")):
        raise ValueError("Eigenvalue enclosure is incompatible with a physical block.")
    value = value.intersection(arb("0.5", "0.5"))
    if value > 0:
        return -value * value.log() / arb(2).log()
    upper = value.upper()
    if upper <= 0:
        return arb(0)
    turning = (-arb(1)).exp()  # h increases on [0,1/e].
    if upper < turning:
        maximum = -upper * upper.log() / arb(2).log()
    else:
        maximum = 1 / (arb(1).exp() * arb(2).log())
    return arb(0).union(maximum)


def entropy_matrix_bound(matrix: acb_mat, *, algorithm: str | None = None) -> tuple[arb, dict]:
    """Certified spectrum, with multiplicities, followed by entropy bounds."""
    if algorithm == "approx":
        raise ValueError("Approximate eigenvalues are forbidden for certification.")
    eigenvalues = matrix.eig(multiple=True, algorithm=algorithm)
    result = arb(0)
    zero_crossings = 0
    smallest_lower = None
    largest_radius = arb(0)
    for eigenvalue in eigenvalues:
        if not eigenvalue.is_finite() or not eigenvalue.imag.contains(0):
            raise ValueError("Failed to enclose real eigenvalues of the Hermitian block.")
        value = eigenvalue.real
        if not value > 0:
            zero_crossings += 1
        lower = value.lower()
        smallest_lower = lower if smallest_lower is None else lower.min(smallest_lower)
        largest_radius = largest_radius.max(value.rad())
        result += entropy_eigenvalue_bound(value)
    return result, {
        "dimension": len(eigenvalues),
        "zero_crossing_eigenvalue_intervals": zero_crossings,
        "minimum_eigenvalue_lower_bound": str(smallest_lower.fmpq()) if smallest_lower is not None else "0",
        "maximum_eigenvalue_radius": largest_radius.str(12, more=True),
    }


class ArbSymmetricEvaluator:
    """All-sector kernel at one exact noise value and a fixed bit precision.

    A cutoff limits input excitation only. Output blocks always retain their
    full (n-2q+1) dimension. Methods must be called at the construction
    precision (evaluate checks this); use one instance per workprec context.
    """

    def __init__(self, n: int, p_total: str | Fraction, *, cutoff: int | None = None,
                 precision_bits: int = 192, progress: bool = False):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        self.n = n
        self.cutoff = n if cutoff is None else min(int(cutoff), n)
        if self.cutoff < 0:
            raise ValueError("cutoff must be nonnegative.")
        self.p_total = _fraction(p_total)
        if not 0 <= self.p_total <= Fraction(3, 4):
            raise ValueError("This positive-kernel implementation supports total p in [0,3/4].")
        self.precision_bits = int(precision_bits)
        if flint.ctx.prec != self.precision_bits:
            raise ValueError("Construct and evaluate inside flint.ctx.workprec(precision_bits).")
        self.progress = progress
        self.sectors = []
        start = time.perf_counter()
        self._build()
        self.build_seconds = time.perf_counter() - start

    def _build(self):
        n, cutoff = self.n, self.cutoff
        noise = self.p_total / 3
        r = arb(fmpq(noise.numerator, noise.denominator))
        a, b, eta = 1 - 2*r, 2*r, 1 - 4*r
        determinant = 2*r*(1 - 2*r)
        ap = [arb_poly([a, b])**s for s in range(n + 1)]
        dp = [arb_poly([b, a])**s for s in range(n + 1)]
        eta_powers = [eta**s for s in range(2*n + 1)]
        pair_products = {}
        for q in range(n//2 + 1):
            started = time.perf_counter()
            d = n - 2*q
            multiplicity = comb(n, q) - (comb(n, q - 1) if q else 0)
            detpoly = arb_poly([1, 1])**(2*q) * (determinant**q * multiplicity)
            products = {}
            diagonals = []
            for delta in range(min(d, cutoff) + 1):
                invnorm = [1 / arb(comb(n, k)*comb(n, k-delta)).sqrt()
                           for k in range(delta, cutoff + 1)]
                kernel = []
                for j in range(d + 1 - delta):
                    i = j + delta
                    u, v = d-j, d-i
                    coefficient = arb_poly([])
                    for s in range(max(0, u+v-d), min(u, v)+1):
                        bb, cc = v-s, u-s
                        e = d-u-v+s
                        pair = (s, e)
                        if pair not in pair_products:
                            pair_products[pair] = ap[s] * dp[e]
                        if pair not in products:
                            products[pair] = pair_products[pair] * detpoly
                        factor = comb(v, s) * comb(d-v, u-s) * eta_powers[bb+cc]
                        coefficient += products[pair].left_shift(cc) * factor
                    normalization = arb(fmpq(comb(d, i), comb(d, j))).sqrt()
                    kernel.append([coefficient[k] * normalization * invnorm[k-delta]
                                   for k in range(delta, cutoff + 1)])
                diagonals.append(kernel)
            self.sectors.append({"q": q, "d": d, "multiplicity": multiplicity,
                                 "diagonals": diagonals})
            if self.progress:
                print(json.dumps({"stage": "arb_kernel", "n": n, "q": q,
                                  "dimension": d+1, "seconds": time.perf_counter()-started}), flush=True)

    def evaluate(self, coefficients: np.ndarray, *, eig_algorithm: str | None = None) -> dict:
        if flint.ctx.prec != self.precision_bits:
            raise ValueError("The active precision differs from the kernel construction precision.")
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.ndim != 2 or coefficients.shape[1] not in (self.n+1, self.cutoff+1):
            raise ValueError("coefficients must have shape (rank,n+1) or (rank,cutoff+1).")
        if not np.all(np.isfinite(coefficients)) or not coefficients.shape[0]:
            raise ValueError("Finite nonempty coefficients required.")
        if coefficients.shape[1] > self.cutoff+1:
            if np.any(coefficients[:, self.cutoff+1:] != 0):
                raise ValueError("Nonzero input coefficient beyond cutoff.")
            coefficients = coefficients[:, :self.cutoff+1]
        rank = coefficients.shape[0]
        norm_exact = fmpq(0)
        psi = []
        for row in coefficients:
            converted = []
            for value in row:
                real, imag = exact_float(value.real), exact_float(value.imag)
                norm_exact += real*real + imag*imag
                converted.append(acb(arb(real), arb(imag)))
            psi.append(converted)
        if norm_exact <= 0:
            raise ValueError("Zero input state.")
        norm = arb(norm_exact)
        products = {}
        for delta in range(self.cutoff+1):
            products[delta] = [[[psi[r][k]*psi[c][k-delta].conjugate()/norm
                                  for k in range(delta, self.cutoff+1)]
                                 for c in range(rank)] for r in range(rank)]
        total_ic, total_trace = arb(0), acb(0)
        sector_results = []
        started = time.perf_counter()
        for sector in self.sectors:
            sector_start = time.perf_counter()
            q, d = sector["q"], sector["d"]
            size = d+1
            joint = acb_mat(rank*size, rank*size)
            for delta, kernel in enumerate(sector["diagonals"]):
                row_products = products[delta]
                for r in range(rank):
                    for c in range(rank):
                        for j, values in enumerate(kernel):
                            i = j+delta
                            value = acb(0)
                            for k_index, kernel_value in enumerate(values):
                                value += row_products[r][c][k_index] * kernel_value
                            joint[r*size+i, c*size+j] = value
                            if delta:
                                joint[c*size+j, r*size+i] = value.conjugate()
            output = acb_mat(size, size)
            for i in range(size):
                for j in range(size):
                    output[i, j] = sum((joint[r*size+i, r*size+j] for r in range(rank)), acb(0))
            weight = sum((output[i, i] for i in range(size)), acb(0))
            total_trace += weight
            joint_entropy, joint_diagnostics = entropy_matrix_bound(joint, algorithm=eig_algorithm)
            output_entropy, output_diagnostics = entropy_matrix_bound(output, algorithm=eig_algorithm)
            contribution = output_entropy-joint_entropy
            total_ic += contribution
            sector_results.append({
                "q": q, "dimension": size, "multiplicity": sector["multiplicity"],
                "weighted_sector_probability": ball_record(weight.real, 22),
                "coherent_information_bits": ball_record(contribution, 22),
                "joint_spectrum": joint_diagnostics, "output_spectrum": output_diagnostics,
                "seconds": time.perf_counter()-sector_start,
            })
            if self.progress:
                print(json.dumps({"stage": "arb_eigenvalues", "n": self.n, "q": q,
                                  "ic_ball": contribution.str(12, more=True),
                                  "seconds": time.perf_counter()-sector_start}), flush=True)
        if not total_trace.real.contains(1) or not total_trace.imag.contains(0):
            raise ValueError("All-sector trace enclosure does not contain one.")
        if not total_ic.is_finite():
            raise ValueError("Coherent-information interval is not finite.")
        digits = min(60, ceil(self.precision_bits*log10(2)))
        sign = "positive" if total_ic > 0 else "negative" if total_ic < 0 else "unresolved"
        return {
            "n": self.n, "rank_rows": rank, "input_cutoff": self.cutoff,
            "p_total_exact": str(self.p_total), "p_per_pauli_exact": str(self.p_total/3),
            "coefficient_semantics": "exact IEEE754 binary64 dyadics, total squared norm normalized exactly",
            "coefficient_bytes_sha256": hashlib.sha256(np.asarray(coefficients, dtype='<c16').tobytes(order='C')).hexdigest(),
            "exact_squared_norm": str(norm_exact),
            "precision_bits": self.precision_bits,
            "python_flint_version": flint.__version__,
            "eigenvalue_method": "acb_mat.eig(multiple=True), certified ball algorithm; never approx",
            "physical_spectral_constraint": "each multiplicity-weighted block has spectrum in [0,1] by the exact channel formula",
            "coherent_information_bits": ball_record(total_ic, digits),
            "achievable_rate_lower_bound_bits_per_channel_use": ball_record(total_ic/self.n, digits),
            "weighted_total_trace": ball_record(total_trace.real, digits),
            "sign": sign,
            "certified_positive_coherent_information": bool(total_ic > 0),
            "all_spin_sectors_included": True,
            "build_seconds": self.build_seconds,
            "evaluation_seconds": time.perf_counter()-started,
            "sectors": sector_results,
            "evidence_status": "rigorous_ball_arithmetic_enclosure_with_FLINT_Arb_trusted",
            "scope": "fixed witness sign only; not optimality, not a capacity converse, not the paper's integer certificate",
        }


def load_coefficients(path: Path, n: int, key: str = "purification") -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError(f"Missing key {key}; archive keys are {list(archive.keys())}.")
            coefficients = archive[key]
    elif path.suffix == ".npy":
        coefficients = np.load(path, allow_pickle=False)
    else:
        raw = np.fromfile(path, dtype="<c16")
        if raw.size % (n+1):
            raise ValueError("Raw payload length is not a multiple of n+1 complex128 coefficients.")
        coefficients = raw.reshape(-1, n+1)
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[1] != n+1:
        raise ValueError("Payload must include the full n+1 ascending Dicke coordinates.")
    return coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p-total", type=str, required=True)
    parser.add_argument("--key", default="purification")
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--precision-bits", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    coefficients = load_coefficients(args.payload, args.n, args.key)
    with flint.ctx.workprec(args.precision_bits):
        model = ArbSymmetricEvaluator(args.n, args.p_total, cutoff=args.cutoff,
                                       precision_bits=args.precision_bits, progress=args.progress)
        report = model.evaluate(coefficients)
    report["payload_path"] = str(args.payload.resolve())
    report["payload_file_sha256"] = hashlib.sha256(args.payload.read_bytes()).hexdigest()
    report["evaluator_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sectors"}, indent=2))


if __name__ == "__main__":
    main()
