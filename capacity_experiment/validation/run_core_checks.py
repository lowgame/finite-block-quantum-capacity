"""Independent tests of the main symmetric evaluator, including gradients.

Only reads core.py. All optional outputs are written into validation/.
Complex gradient convention tested: df = Re(vdot(gradient, direction)).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True  # Importing core must not create files outside validation/.

import numpy as np

try:
    from .dense_oracle import audit_symmetric_blocks, dense_coherent_information_dicke
except ImportError:
    from dense_oracle import audit_symmetric_blocks, dense_coherent_information_dicke


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("symmetric_core_under_test", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load core from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(core_path: Path, seed: int = 731061, max_n: int = 5) -> dict:
    core = load_core(core_path)
    rng = np.random.default_rng(seed)
    cases = []
    gradient_checks = []
    p_values = [0.0, 0.12, 0.194868, 0.60, 0.7499]
    for n in range(1, max_n + 1):
        random = rng.normal(size=(2, n + 1)) + 1j * rng.normal(size=(2, n + 1))
        random[1] *= 0.43
        pure = rng.normal(size=(1, n + 1)) + 1j * rng.normal(size=(1, n + 1))
        repetition = np.zeros((2, n + 1), dtype=complex)
        repetition[0, 0] = 1 / np.sqrt(2)
        repetition[1, n] = 1 / np.sqrt(2)
        states = {"random_complex_unequal_rows": random, "pure": pure,
                  "redundant_rank_two_pure": np.vstack([pure, 2 * pure]),
                  "repetition": repetition}

        for p_total in p_values:
            model = core.SymmetricDepolarizing(n, p_total)
            for name, psi in states.items():
                raw_blocks = model.blocks(psi, weighted=False)
                blocks = {n - q: block for q, block in enumerate(raw_blocks)}
                audit = audit_symmetric_blocks(psi, p_total / 3, blocks)
                evaluated = model.evaluate(psi, diagnostics=True)
                audit.update({"state": name, "p_total": p_total,
                              "evaluate_ic_bits": float(evaluated["ic_bits"]),
                              "evaluate_ic_abs_error": abs(evaluated["ic_bits"] - audit["dense_coherent_information_bits"]),
                              "evaluate_trace_error": abs(evaluated["trace"] - 1),
                              "evaluate_negative_eigenvalue_mass": float(evaluated["negative_eigenvalue_mass"])})
                audit["passed"] = bool(audit["passed"] and audit["evaluate_ic_abs_error"] <= 2e-10
                                       and audit["evaluate_trace_error"] <= 2e-10)
                cases.append(audit)

        # The derivative check stays away from singular channel endpoints.
        # Directional derivatives of the *dense* function are independent of
        # the core's assembly and entropy-gradient code.
        for p_total in [0.12, 0.194868]:
            model = core.SymmetricDepolarizing(n, p_total)
            psi = random / np.linalg.norm(random)
            calculated = model.evaluate(psi, gradient=True)
            gradient = calculated["gradient"]
            for label, direction in [
                ("complex", rng.normal(size=psi.shape) + 1j * rng.normal(size=psi.shape)),
                ("real", rng.normal(size=psi.shape).astype(complex)),
                ("imaginary", 1j * rng.normal(size=psi.shape)),
            ]:
                direction /= np.linalg.norm(direction)
                predicted = float(np.vdot(gradient, direction).real)
                steps = [2e-4, 5e-5, 1e-5]
                estimates = []
                for h in steps:
                    plus = dense_coherent_information_dicke(psi + h * direction, p_total / 3).coherent_information
                    minus = dense_coherent_information_dicke(psi - h * direction, p_total / 3).coherent_information
                    estimate = (plus - minus) / (2 * h)
                    estimates.append({"step": h, "dense_derivative": estimate,
                                      "absolute_error": abs(estimate - predicted)})
                best_error = min(e["absolute_error"] for e in estimates)
                gradient_checks.append({"n": n, "p_total": p_total, "direction": label,
                                        "predicted_derivative": predicted, "estimates": estimates,
                                        "best_absolute_error": best_error, "tolerance": 2e-6,
                                        "passed": best_error <= 2e-6})

            for label, direction in [("normalization_radial", psi), ("global_phase", 1j * psi)]:
                error = abs(float(np.vdot(gradient, direction).real))
                gradient_checks.append({"n": n, "p_total": p_total, "direction": label,
                                        "best_absolute_error": error, "tolerance": 2e-9,
                                        "passed": error <= 2e-9})

        if n >= 2:
            cutoff = n - 1
            truncated = random.copy()
            truncated[:, cutoff + 1:] = 0
            model = core.SymmetricDepolarizing(n, 0.194868, cutoff=cutoff)
            blocks = {n - q: block for q, block in enumerate(model.blocks(truncated, weighted=False))}
            audit = audit_symmetric_blocks(truncated, 0.194868 / 3, blocks)
            audit.update({"state": "input_cutoff_keeps_all_output_sectors", "cutoff": cutoff,
                          "p_total": 0.194868})
            cases.append(audit)

    return {
        "core_path": str(core_path), "seed": seed, "n_range": [1, max_n],
        "num_dense_cases": len(cases),
        "num_gradient_checks": len(gradient_checks),
        "failed_dense_cases": sum(not c["passed"] for c in cases),
        "failed_gradient_checks": sum(not c["passed"] for c in gradient_checks),
        "maximum_dense_ic_error": max(c["errors"]["coherent_information_abs"] for c in cases),
        "maximum_block_entry_error": max(c["errors"]["block_entry_max_abs"] for c in cases),
        "maximum_gradient_best_error": max(c["best_absolute_error"] for c in gradient_checks),
        "passed": all(c["passed"] for c in cases) and all(c["passed"] for c in gradient_checks),
        "cases": cases, "gradient_checks": gradient_checks,
        "evidence_status": "independent_floating_point_validation_not_certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=Path(__file__).resolve().parents[1] / "core.py")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "core_check_results.json")
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()
    start = time.perf_counter()
    report = run(args.core, max_n=args.max_n)
    report["elapsed_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "gradient_checks")}, indent=2))
    if not report["passed"]:
        print(json.dumps({"dense_failures": [c for c in report["cases"] if not c["passed"]],
                          "gradient_failures": [c for c in report["gradient_checks"] if not c["passed"]]}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
