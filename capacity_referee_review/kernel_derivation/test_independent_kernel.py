"""Independent exact tensor-polynomial and legacy-block cross-checks.

The tensor oracle enumerates computational-basis strings and signed singlet
vectors.  It contains no binomial coefficient-extraction kernel.  Polynomial
equality is checked over the integers, not by evaluating at sample points.

The legacy-block comparison invokes its original block-construction path with
an explicitly inert entropy callback that records matrices; no sign claim is
made by that callback.  Every resulting joint/output entry is then compared
with the new independent integer-column construction.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from math import comb
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True

from independent_kernel import (
    ExactColumnPolynomials, iter_weighted_blocks, flint, arb, fmpz_poly, np
)

ROOT = Path(__file__).resolve().parents[2]


def singlet_dicke_raw(n, q, i):
    """Signed unnormalized sparse vector; squared norm is 2**q*C(d,i)."""
    prefixes = [(0, 1)]
    for _ in range(q):
        prefixes = [(x*4+bits, sign*sgn)
                    for x, sign in prefixes for bits, sgn in ((1, 1), (2, -1))]
    d = n-2*q
    tails = [x for x in range(2**d) if x.bit_count() == i]
    return [(prefix*2**d+tail, sign) for prefix, sign in prefixes for tail in tails]


def exact_tensor_polynomial_check(max_n=6):
    rows = []
    for p in (Fraction(0), Fraction(3, 4), Fraction(79, 400), Fraction(1949, 10000)):
        for n in range(1, max_n+1):
            start = time.perf_counter()
            kernel = ExactColumnPolynomials(n, p)
            single = (
                (fmpz_poly([kernel.a, kernel.b]), fmpz_poly([kernel.e])),
                (fmpz_poly([0, kernel.e]), fmpz_poly([kernel.b, kernel.a])),
            )
            # Direct tensor product in the computational basis, with t formal.
            tensor = []
            for x in range(2**n):
                row = []
                for y in range(2**n):
                    value = fmpz_poly([1])
                    for bit in range(n):
                        value *= single[(x >> bit) & 1][(y >> bit) & 1]
                    row.append(value)
                tensor.append(row)
            matrices, entries, coefficients = 0, 0, 0
            for q in range(n//2+1):
                d = n-2*q
                vectors = [singlet_dicke_raw(n, q, i) for i in range(d+1)]
                for i in range(d+1):
                    for j in range(d+1):
                        direct = fmpz_poly([])
                        for x, sx in vectors[i]:
                            for y, sy in vectors[j]:
                                direct += tensor[x][y] * (sx*sy)
                        column = kernel.sector_column_entry(q, i, j)
                        expected = column * (2**q * comb(d, j))
                        if direct != expected:
                            raise AssertionError(("exact polynomial mismatch", n, str(p), q, i, j,
                                                  str(direct), str(expected)))
                        entries += 1
                        coefficients += n+1
                matrices += 1
            rows.append({"n": n, "p_total_exact": str(p), "sectors": matrices,
                         "matrix_entries": entries, "integer_coefficients_checked": coefficients,
                         "seconds": time.perf_counter()-start, "all_equal": True})
    return {"method": "exact integer polynomials from signed computational-basis tensor products",
            "maximum_n": max_n, "cases": rows,
            "matrix_entries": sum(r["matrix_entries"] for r in rows),
            "integer_coefficients_checked": sum(r["integer_coefficients_checked"] for r in rows),
            "all_equal": True}


def legacy_block_check(n, p, payload, cutoff, precision=256):
    spec = importlib.util.spec_from_file_location(
        "legacy_interval_for_matrix_capture", ROOT / "capacity_experiment/validation/interval_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = Path(payload).read_bytes()
    coefficients = np.frombuffer(raw, dtype="<c16").reshape(-1, n+1).copy()
    captures = []
    def capture(matrix, *, algorithm=None):
        captures.append(matrix)
        return arb(0), {"matrix_capture_only": True}
    module.entropy_matrix_bound = capture
    started = time.perf_counter()
    with flint.ctx.workprec(precision):
        legacy = module.ArbSymmetricEvaluator(n, p, cutoff=cutoff, precision_bits=precision)
        legacy.evaluate(coefficients)  # Its entropy callback records matrices only.
        if len(captures) != 2*(n//2+1):
            raise AssertionError("Unexpected number of legacy matrices.")
        compared = 0
        maximum_midpoint_difference = arb(0)
        total_trace = arb(0)
        sector_rows = []
        for q, joint, output in iter_weighted_blocks(n, p, coefficients, cutoff=cutoff,
                                                     precision_bits=precision):
            sector_entries = 0
            for new, old in ((joint, captures[2*q]), (output, captures[2*q+1])):
                if new.nrows() != old.nrows() or new.ncols() != old.ncols():
                    raise AssertionError("Matrix dimensions differ.")
                for i in range(new.nrows()):
                    for j in range(new.ncols()):
                        if not new[i, j].overlaps(old[i, j]):
                            raise AssertionError(("nonoverlapping entries", n, str(p), q, i, j,
                                                  str(new[i,j]), str(old[i,j])))
                        delta = abs(new[i,j].mid() - old[i,j].mid())
                        maximum_midpoint_difference = maximum_midpoint_difference.max(delta)
                        compared += 1
                        sector_entries += 1
            weight = sum((output[i,i].real for i in range(output.nrows())), arb(0))
            total_trace += weight
            sector_rows.append({"q": q, "joint_dimension": joint.nrows(),
                                "output_dimension": output.nrows(), "entries_compared": sector_entries,
                                "all_entry_intervals_overlap": True})
        if not total_trace.contains(1):
            raise AssertionError("Independent blocks do not enclose unit total trace.")
        return {"n": n, "p_total_exact": str(p), "cutoff": cutoff,
                "precision_bits": precision, "payload_sha256": hashlib.sha256(raw).hexdigest(),
                "all_entry_intervals_overlap": True, "entries_compared": compared,
                "max_midpoint_difference": maximum_midpoint_difference.str(18, more=True),
                "total_trace": total_trace.str(18, more=True), "sectors": sector_rows,
                "seconds": time.perf_counter()-started,
                "warning": "The inert legacy entropy callback was used solely for matrix capture, not certification."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--legacy-n45", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"scope": "new internal algebra and kernel checks, not external replication"}
    if not args.skip_exact:
        report["exact_tensor_polynomials"] = exact_tensor_polynomial_check(args.max_n)
        print(json.dumps({"exact_tensor_polynomials": "PASS",
                          "matrix_entries": report["exact_tensor_polynomials"]["matrix_entries"]}), flush=True)
    if args.legacy_n45:
        report["full_n45_block_comparison"] = legacy_block_check(
            45, Fraction(1949,10000), ROOT / "capacity_experiment/results/A45_continued.c128", 45)
        print(json.dumps({"full_n45_block_comparison": "PASS",
                          "entries_compared": report["full_n45_block_comparison"]["entries_compared"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
