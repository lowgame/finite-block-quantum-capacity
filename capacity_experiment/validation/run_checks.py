"""Run meaningful analytic and independent dense-oracle checks.

Example:
  python capacity_experiment/validation/run_checks.py --output capacity_experiment/validation/check_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    from .dense_oracle import (
        audit_symmetric_blocks, dense_coherent_information,
        dense_coherent_information_dicke, dense_schur_blocks,
        depolarize_affine, depolarize_pauli, dicke_isometry,
        entropy_bits, maximally_mixed_one_qubit_formula,
        schur_sectors, singlet_dicke_embedding,
    )
except ImportError:
    from dense_oracle import (
        audit_symmetric_blocks, dense_coherent_information,
        dense_coherent_information_dicke, dense_schur_blocks,
        depolarize_affine, depolarize_pauli, dicke_isometry,
        entropy_bits, maximally_mixed_one_qubit_formula,
        schur_sectors, singlet_dicke_embedding,
    )


def run_checks(seed: int = 20261011, max_n: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    checks = []

    def record(name: str, error: float, tolerance: float = 2e-10, **extra):
        checks.append({"name": name, "error": float(error), "tolerance": tolerance,
                       "passed": bool(np.isfinite(error) and error <= tolerance), **extra})

    p_values = [0.0, 0.01, 16239 / 250000, 0.12, 0.25, 1 / 3]
    for p in p_values:
        result = dense_coherent_information(np.eye(2) / 2, 1, p)
        record("maximally_mixed_one_qubit", abs(result.coherent_information - maximally_mixed_one_qubit_formula(p)),
               p_per_pauli=p)

    if not 1 <= max_n <= 6:
        raise ValueError("max_n must be between 1 and 6.")
    for n in range(1, max_n + 1):
        w = dicke_isometry(n)
        record("dicke_isometry", np.max(np.abs(w.conj().T @ w - np.eye(n + 1))), n=n)
        sectors = schur_sectors(n)
        record("schur_multiplicity_dimension", abs(sum(s["dimension"] * s["multiplicity"] for s in sectors) - 2**n),
               tolerance=0.0, n=n)
        for sector in sectors:
            emb = singlet_dicke_embedding(n, sector["f"])
            record("sector_isometry", np.max(np.abs(emb.conj().T @ emb - np.eye(sector["dimension"]))), n=n, f=sector["f"])

        coefficients = rng.normal(size=(2, n + 1)) + 1j * rng.normal(size=(2, n + 1))
        coefficients[1] *= 0.37  # Unequal norms are intentional; no per-row normalization.
        pure = rng.normal(size=(1, n + 1)) + 1j * rng.normal(size=(1, n + 1))
        redundant_pure = np.vstack([pure, 2 * pure])  # Power-of-two scaling is exact in binary64.

        for p in p_values:
            pauli = dense_coherent_information_dicke(coefficients, p, method="pauli")
            affine = dense_coherent_information_dicke(coefficients, p, method="affine")
            record("pauli_vs_affine_joint_channel", np.max(np.abs(pauli.joint_output - affine.joint_output)), n=n, p_per_pauli=p)
            record("pauli_vs_affine_coherent_information", abs(pauli.coherent_information - affine.coherent_information), n=n, p_per_pauli=p)
            for name, value in pauli.diagnostics.items():
                if "min_eigenvalue" in name:
                    record(name, max(0, -value), n=n, p_per_pauli=p)
                else:
                    record(name, value, n=n, p_per_pauli=p)
            for coefficients_pure in (pure, redundant_pure):
                pure_result = dense_coherent_information_dicke(coefficients_pure, p)
                record("pure_input_zero_coherent_information", abs(pure_result.coherent_information), n=n,
                       reference_dim=coefficients_pure.shape[0], p_per_pauli=p)

            blocks = dense_schur_blocks(coefficients, p)
            audit = audit_symmetric_blocks(coefficients, p, blocks)
            record("schur_full_spectra_multiplicity_entropy", max(audit["errors"].values()), n=n, p_per_pauli=p)
            record("coherent_information_input_entropy_bound", max(0, abs(pauli.coherent_information) - pauli.input_entropy), n=n, p_per_pauli=p)
            if p == 0:
                record("identity_channel_coherent_information", abs(pauli.coherent_information - pauli.input_entropy), n=n)
                for f, block in blocks.items():
                    if f < n:
                        record("identity_channel_lower_sector_zero", np.max(np.abs(block)), n=n, f=f)
            if p == 0.25:
                record("completely_depolarizing_coherent_information", abs(pauli.coherent_information + pauli.input_entropy), n=n)
                record("completely_depolarizing_output", np.max(np.abs(pauli.output_state - np.eye(2**n) / 2**n)), n=n)

        # Check the channel on a non-Hermitian matrix unit, so accidental
        # conjugation/transposition is not masked by a real or diagonal input.
        unit = np.outer(w[:, 0], w[:, n].conj())
        a = depolarize_affine(unit, n, 0.064956)
        b = depolarize_pauli(unit, n, 0.064956)
        record("nonhermitian_matrix_unit_channel", np.max(np.abs(a - b)), n=n)

        values = [dense_coherent_information_dicke(coefficients, p).coherent_information
                  for p in np.linspace(0, 0.25, 9)]
        record("fixed_input_data_processing_monotonicity", max(0, float(np.max(np.diff(values)))), n=n)

    # An entropy routine that normalizes each block would return the wrong value.
    subnormalized = np.diag([0.1, 0.2])
    expected = -0.1 * np.log2(0.1) - 0.2 * np.log2(0.2)
    record("subnormalized_entropy_no_renormalization", abs(entropy_bits(subnormalized) - expected))
    rejected_psd = False
    try:
        entropy_bits(np.diag([-1e-4, 1.0001]))
    except ValueError:
        rejected_psd = True
    record("material_psd_violation_rejected", float(not rejected_psd), tolerance=0.0)

    return {
        "seed": seed,
        "n_range": [1, max_n],
        "channel_convention": "per-Pauli p; total nonidentity error=3p; contraction eta=1-4p",
        "num_checks": len(checks),
        "num_failed": sum(not c["passed"] for c in checks),
        "maximum_absolute_check_error": max(c["error"] for c in checks),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "evidence_status": "floating_point_cross_check_not_certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()
    start = time.perf_counter()
    report = run_checks(args.seed, args.max_n)
    report["elapsed_seconds"] = time.perf_counter() - start
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: v for k, v in report.items() if k != "checks"}
    print(json.dumps(summary, indent=2))
    if not report["passed"]:
        print(json.dumps([c for c in report["checks"] if not c["passed"]], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
