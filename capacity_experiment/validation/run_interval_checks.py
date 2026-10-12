"""Small exact identities and independent dense checks for interval_core.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

try:
    from .interval_core import ArbSymmetricEvaluator, flint, arb, acb_mat, entropy_matrix_bound
    from .dense_oracle import dense_coherent_information_dicke
except ImportError:
    from interval_core import ArbSymmetricEvaluator, flint, arb, acb_mat, entropy_matrix_bound
    from dense_oracle import dense_coherent_information_dicke


def run() -> dict:
    checks = []
    rng = np.random.default_rng(709341)
    with flint.ctx.workprec(160):
        # Repeated eigenvalues, exact zeros, and a zero-crossing enclosure.
        h, diagnostics = entropy_matrix_bound(acb_mat([[0, 0, 0], [0, arb("1/2"), 0], [0, 0, arb("1/2")]]))
        checks.append({"name": "entropy_zeros_and_multiplicity", "passed": h.contains(1),
                       "entropy_ball": h.str(35, more=True), "diagnostics": diagnostics})
        for p_string in ["0", "0.194868", "0.75"]:
            model = ArbSymmetricEvaluator(1, p_string, precision_bits=160)
            result = model.evaluate(np.eye(2, dtype=complex))
            p = arb(p_string) / 3
            probabilities = [1-3*p, p, p, p]
            exact_formula = arb(1)
            for probability in probabilities:
                if probability > 0:
                    exact_formula += probability * probability.log() / arb(2).log()
            value = arb(result["coherent_information_bits"]["ball"])
            checks.append({"name": "maximally_mixed_formula", "p_total_exact": p_string,
                           "passed": value.overlaps(exact_formula),
                           "computed_ball": value.str(35, more=True),
                           "analytic_ball": exact_formula.str(35, more=True)})

        for n in range(1, 6):
            pure = rng.normal(size=(1, n+1)) + 1j*rng.normal(size=(1, n+1))
            random = rng.normal(size=(2, n+1)) + 1j*rng.normal(size=(2, n+1))
            model = ArbSymmetricEvaluator(n, "0.1955", precision_bits=160)
            for name, psi in [("pure", pure), ("exact_redundant_pure", np.vstack([pure, 2*pure]))]:
                result = model.evaluate(psi)
                value = arb(result["coherent_information_bits"]["ball"])
                checks.append({"name": "pure_zero_ic", "representation": name, "n": n,
                               "passed": value.contains(0), "ball": value.str(35, more=True)})
            result = model.evaluate(random)
            value = arb(result["coherent_information_bits"]["ball"])
            dense = dense_coherent_information_dicke(random, 0.1955/3).coherent_information
            discrepancy = abs(float(value.mid())-dense)
            checks.append({"name": "random_complex_vs_dense", "n": n,
                           "passed": discrepancy <= 2e-10, "difference": discrepancy})
            scaled = model.evaluate(2*random)
            scaled_value = arb(scaled["coherent_information_bits"]["ball"])
            checks.append({"name": "exact_global_scale_invariance", "n": n,
                           "passed": scaled_value.overlaps(value)})

        n, cutoff = 5, 2
        psi = rng.normal(size=(2,n+1)) + 1j*rng.normal(size=(2,n+1))
        psi[:,cutoff+1:] = 0
        model = ArbSymmetricEvaluator(n, "0.1955", cutoff=cutoff, precision_bits=160)
        result = model.evaluate(psi)
        value = arb(result["coherent_information_bits"]["ball"])
        dense = dense_coherent_information_dicke(psi, 0.1955/3).coherent_information
        discrepancy = abs(float(value.mid())-dense)
        checks.append({"name": "input_cutoff_full_output_sectors", "n": n, "cutoff": cutoff,
                       "passed": discrepancy <= 2e-10 and len(result["sectors"]) == n//2+1,
                       "difference": discrepancy})

    return {"passed": all(c["passed"] for c in checks), "num_checks": len(checks),
            "num_failed": sum(not c["passed"] for c in checks), "checks": checks,
            "precision_bits": 160, "python_flint_version": flint.__version__,
            "scope": "validation of formulas and software, separate from each fixed-witness interval bound",
            "interval_source_sha256": hashlib.sha256(Path(__file__).with_name('interval_core.py').read_bytes()).hexdigest()}


if __name__ == "__main__":
    start = time.perf_counter()
    report = run()
    report["elapsed_seconds"] = time.perf_counter()-start
    path = Path(__file__).with_name("interval_check_results.json")
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "checks"},indent=2))
    if not report["passed"]:
        print(json.dumps([c for c in report["checks"] if not c["passed"]],indent=2))
        raise SystemExit(1)
