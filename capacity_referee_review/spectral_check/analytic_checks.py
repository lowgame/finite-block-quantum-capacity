"""Check the independent spectral method against prescribed exact spectra."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from capacity_referee_review.spectral_check import residual_verifier as rv


def prescribed_matrix(spectrum, complex_rotation=False):
    n = len(spectrum)
    # A rational Householder unitary. Its exact spectrum is the supplied D;
    # only its rigorous entrywise representation is rounded to Arb balls.
    v = [rv.acb(i+1, (i%2) if complex_rotation else 0) for i in range(n)]
    squared_norm = sum((abs(z)**2 for z in v), rv.arb(0))
    u = rv.acb_mat(n, n)
    for i in range(n):
        for j in range(n):
            u[i, j] = (1 if i == j else 0) - 2*v[i]*v[j].conjugate()/squared_norm
    d = rv.acb_mat(n, n)
    for i, value in enumerate(spectrum):
        d[i, i] = rv.rational(value)
    return u*d*u.conjugate().transpose()


def main():
    tests = [
        ("one-dimensional pure", [Fraction(1)], False),
        ("two-dimensional mixed", [Fraction(1,5),Fraction(4,5)], False),
        ("repeated positive eigenvalues", [Fraction(1,2),Fraction(1,2)], True),
        ("complex rotation with exact zero", [Fraction(0),Fraction(1,2)], True),
        ("three-dimensional unequal spectrum", [Fraction(1,7),Fraction(2,7),Fraction(4,7)], True),
        ("three-dimensional rank deficiency", [Fraction(0),Fraction(1,3),Fraction(2,3)], False),
        ("four-dimensional separated tiny eigenvalue", [Fraction(1,10**40),Fraction(1,9),Fraction(2,9),Fraction(1,3)], True),
        ("five-dimensional subnormalized spectrum", [Fraction(k,30) for k in range(1,6)], True),
    ]
    report = []
    for name, spectrum, complex_rotation in tests:
        with rv.flint.ctx.workprec(256):
            matrix = prescribed_matrix(spectrum, complex_rotation)
            value, diagnostics = rv.residual_entropy(matrix)
        with rv.flint.ctx.workprec(768):
            exact_entropy_enclosure = rv.arb(0)
            for probability in spectrum:
                if probability:
                    x = rv.rational(probability)
                    exact_entropy_enclosure -= x*x.log()/rv.arb(2).log()
            passed = value.contains(exact_entropy_enclosure)
            if not passed:
                raise AssertionError(f"Known-spectrum entropy not contained: {name}")
            report.append({
                "name": name,
                "spectrum": [str(x) for x in spectrum],
                "complex_rotation": complex_rotation,
                "accepted_entropy": rv.record(value),
                "known_spectrum_entropy": rv.record(exact_entropy_enclosure),
                "contains_high_precision_reference": passed,
                "components": diagnostics["component_dimensions"],
            })
    # This input interval straddles the maximum of h(x). A midpoint-only or
    # endpoint-only upper bound would be invalid; the peak must be included.
    with rv.flint.ctx.workprec(256):
        scalar_input = rv.arb(rv.fmpq(1,3), rv.fmpq(1,10))
        scalar_bound, _ = rv.residual_entropy(rv.acb_mat([[scalar_input]]))
    with rv.flint.ctx.workprec(768):
        peak = 1/(rv.arb(1).exp()*rv.arb(2).log())
        if not scalar_bound.contains(peak):
            raise AssertionError("Entropy maximum not enclosed.")
        report.append({"name": "uncertain scalar interval includes entropy maximum",
                       "accepted_entropy": rv.record(scalar_bound),
                       "contains_high_precision_reference": True})
    output = {
        "status": "passed",
        "test_count": len(report),
        "evaluation_precision_bits": 256,
        "analytic_reference_precision_bits": 768,
        "reference": "explicit prescribed spectra under rational Householder unitary similarity; scalar entropy maximum",
        "spectral_source_sha256": hashlib.sha256(Path(rv.__file__).read_bytes()).hexdigest(),
        "tests": report,
    }
    destination = Path(__file__).with_name("analytic_check_results.json")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite analytic check record.")
    destination.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"status": output["status"],"test_count": output["test_count"]}))


if __name__ == "__main__":
    main()
