"""Certified interval evaluator with only *provably exact* zero splitting.

Every omitted cross-component entry has zero real and imaginary parts as
exact Arb points. Thus the split is a permutation similarity, not a sparsity
threshold or numerical approximation. Unknown/overlapping-zero balls retain
an edge and can only prevent a split. The underlying kernel is unchanged.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

try:
    from . import interval_core as base
except ImportError:
    import interval_core as base

arb, acb_mat, flint = base.arb, base.acb_mat, base.flint
_unsplit_entropy = base.entropy_matrix_bound


def provably_zero(value) -> bool:
    real, imag = value.real, value.imag
    return (real.is_exact() and imag.is_exact()
            and real.fmpq() == 0 and imag.fmpq() == 0)


def exact_components(matrix) -> list[list[int]]:
    if matrix.nrows() != matrix.ncols():
        raise ValueError("Square matrix required.")
    dimension = matrix.nrows()
    adjacency = [set() for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i+1, dimension):
            if not (provably_zero(matrix[i,j]) and provably_zero(matrix[j,i])):
                adjacency[i].add(j)
                adjacency[j].add(i)
    unseen = set(range(dimension))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], []
        while stack:
            i = stack.pop()
            component.append(i)
            for j in sorted(adjacency[i] & unseen):
                unseen.remove(j)
                stack.append(j)
        components.append(sorted(component))
    return components


def entropy_matrix_bound_split(matrix, *, algorithm=None):
    components = exact_components(matrix)
    entropy = arb(0)
    diagnostics = []
    for component in components:
        submatrix = acb_mat([[matrix[i,j] for j in component] for i in component])
        contribution, diagnostic = _unsplit_entropy(submatrix, algorithm=algorithm)
        entropy += contribution
        diagnostics.append(diagnostic)
    smallest_lower = min(Fraction(d["minimum_eigenvalue_lower_bound"]) for d in diagnostics)
    largest_radius = arb(0)
    for diagnostic in diagnostics:
        largest_radius = largest_radius.max(arb(diagnostic["maximum_eigenvalue_radius"]))
    return entropy, {
        "dimension": matrix.nrows(),
        "component_dimensions": [len(component) for component in components],
        "permutation_components": components,
        "split_criterion": "both off-diagonal entries provably exact zero; no numerical threshold",
        "zero_crossing_eigenvalue_intervals": sum(d["zero_crossing_eigenvalue_intervals"] for d in diagnostics),
        "minimum_eigenvalue_lower_bound": str(smallest_lower),
        "maximum_eigenvalue_radius": largest_radius.str(12, more=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p-total", type=str, required=True)
    parser.add_argument("--key", default="purification")
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--precision-bits", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    coefficients = base.load_coefficients(args.payload, args.n, args.key)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base_hash = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    payload_hash = hashlib.sha256(args.payload.read_bytes()).hexdigest()
    original = base.entropy_matrix_bound
    base.entropy_matrix_bound = entropy_matrix_bound_split
    try:
        with flint.ctx.workprec(args.precision_bits):
            model = base.ArbSymmetricEvaluator(args.n, args.p_total, cutoff=args.cutoff,
                                               precision_bits=args.precision_bits, progress=args.progress)
            report = model.evaluate(coefficients)
    finally:
        base.entropy_matrix_bound = original
    report.update({
        "payload_path": str(args.payload.resolve()),
        "payload_file_sha256": payload_hash,
        "evaluator_source_sha256": base_hash,
        "exact_zero_splitter_source_sha256": source_hash,
        "eigenvalue_method": "acb_mat.eig(multiple=True) on provably exact permutation-direct-sum components",
        "exact_zero_splitter": "conservative graph: every entry not proven exactly zero is an edge; no tolerance",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "sectors"},indent=2))


if __name__ == "__main__":
    main()
