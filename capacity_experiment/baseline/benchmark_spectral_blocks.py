"""Meaningful floating-point checks/benchmarks for exact-zero eigensplitting."""

from pathlib import Path
import hashlib
import io
import json
import sys
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import SymmetricDepolarizing
from spectral_blocks import exact_zero_components, eigh_exact_blocks


def entropy(values):
    positive = values[values > 0]
    return -float(positive @ np.log2(positive))


def check_matrix(matrix, label):
    a = np.asarray(matrix)
    w, v = eigh(a, check_finite=False, driver="evd")
    sw, sv = eigh_exact_blocks(a, check_finite=False, driver="evd")
    only = eigh_exact_blocks(a, eigvals_only=True, check_finite=False, driver="evr")
    ref_only = eigh(a, eigvals_only=True, check_finite=False, driver="evr")
    scale = max(float(np.max(np.abs(w))), np.finfo(float).tiny)
    floor = max(abs(float(np.trace(a).real))*1e-15, 1e-300)
    log_full = (v * np.log2(np.maximum(w, floor))) @ v.conj().T
    log_split = (sv * np.log2(np.maximum(sw, floor))) @ sv.conj().T
    eigen_error = float(np.max(np.abs(w-sw)))
    only_error = float(np.max(np.abs(only-ref_only)))
    reconstruction = np.linalg.norm((sv*sw)@sv.conj().T-a, "fro")
    reconstruction /= max(float(np.linalg.norm(a, "fro")), np.finfo(float).tiny)
    orthogonality = float(np.max(np.abs(sv.conj().T@sv-np.eye(len(a)))))
    log_error = float(np.max(np.abs(log_split-log_full)))
    log_relative = float(np.linalg.norm(log_split-log_full, "fro")) / max(
        float(np.linalg.norm(log_full, "fro")), 1.)
    record = {
        "label": label, "dimension": len(a),
        "component_dimensions": [len(indices) for indices in exact_zero_components(a)],
        "full_eigenvalues_absolute_error": eigen_error,
        "full_eigenvalues_relative_error": eigen_error/scale,
        "values_only_absolute_error": only_error,
        "values_only_relative_error": only_error/scale,
        "reconstruction_relative_frobenius_error": float(reconstruction),
        "orthogonality_maximum_absolute_error": orthogonality,
        "matrix_log_floor": floor,
        "floored_log_maximum_absolute_entry_difference": log_error,
        "floored_log_relative_frobenius_difference": log_relative,
        "entropy_absolute_difference_bits": abs(entropy(w)-entropy(sw)),
        "passed_spectral_checks": bool(eigen_error/scale < 1e-11 and
            only_error/scale < 1e-11 and reconstruction < 1e-11 and orthogonality < 1e-11),
    }
    return record, entropy(w), entropy(sw)


def time_spectra(matrices, *, split, eigvals_only, repeats=3):
    solver = eigh_exact_blocks if split else eigh
    driver = "evr" if eigvals_only else "evd"
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for a in matrices:
            solver(a, eigvals_only=eigvals_only, check_finite=False, driver=driver)
        timings.append(time.perf_counter()-start)
    return {"median_seconds": float(np.median(timings)), "runs_seconds": timings}


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(20261011)
    records = []
    with threadpool_limits(limits=1):
        for dimension in (1, 2, 3, 7, 16, 31):
            x = rng.normal(size=(dimension, dimension))+1j*rng.normal(size=(dimension, dimension))
            a = x@x.conj().T
            a /= np.trace(a).real
            records.append(check_matrix(a, f"dense_random_{dimension}")[0])
        a = np.zeros((15, 15), complex)
        offset = 0
        for width in (1, 2, 4, 8):
            x = rng.normal(size=(width, width))+1j*rng.normal(size=(width, width))
            a[offset:offset+width, offset:offset+width] = x@x.conj().T
            offset += width
        order = rng.permutation(len(a))
        a = a[np.ix_(order, order)]/np.trace(a).real
        records.append(check_matrix(a, "permuted_components_1_2_4_8")[0])
        records.append(check_matrix(np.diag([0., .25, .25, .5]), "degenerate_diagonal_with_zero")[0])
        tiny = np.diag([1., 2., 3.]).astype(complex)
        tiny[0, 1] = tiny[1, 0] = 1e-250
        tiny_components = [len(c) for c in exact_zero_components(tiny)]
        assert tiny_components == [2, 1], tiny_components
        records.append(check_matrix(tiny, "tiny_nonzero_edge_1e-250")[0])
        one_triangle = np.diag([1., 2.]).astype(complex)
        one_triangle[0, 1] = 1e-250
        assert [len(c) for c in exact_zero_components(one_triangle)] == [2]
        assert exact_zero_components(np.zeros((0, 0))) == ()
        try:
            eigh_exact_blocks(np.array([[np.nan]]))
        except ValueError:
            pass
        else:
            raise AssertionError("NaN matrix was not rejected.")

        frozen = []
        specs = [
            (60, .1955, 60, ROOT/"candidates/gaussian_n60_p0p195500_final.npz", "purification", ROOT/"candidates/cache"),
            (120, .197, 72, ROOT/"results/n120_optimized_p0p197.npz", "psi", ROOT/"cache"),
        ]
        for n, p, cutoff, payload, key, cache in specs:
            source_path = payload
            payload_bytes = source_path.read_bytes()
            payload_hash = hashlib.sha256(payload_bytes).hexdigest()
            payload = ROOT/f"baseline/frozen_n{n}_{payload_hash[:12]}.npz"
            payload.write_bytes(payload_bytes)
            with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as archive:
                psi = archive[key]
            model = SymmetricDepolarizing(n, p, cutoff=cutoff, cache_dir=cache)
            matrices, case_records = [], []
            ic_full = ic_split = 0.
            for sector, joint in zip(model.sectors, model.blocks(psi)):
                d = sector.d+1
                joint = (joint+joint.conj().T)*.5
                output = np.einsum("rrij->ij", joint.reshape(2, d, 2, d).transpose(0, 2, 1, 3))
                output = (output+output.conj().T)*.5
                jrecord, jf, js = check_matrix(joint, f"n{n}_q{sector.q}_joint")
                orecord, of, os = check_matrix(output, f"n{n}_q{sector.q}_output")
                ic_full += of-jf
                ic_split += os-js
                matrices.extend([joint, output])
                case_records.extend([jrecord, orecord])
            timings = {}
            for values_only in (True, False):
                mode = "values_only" if values_only else "eigenvalues_and_full_eigenvectors"
                unsplit = time_spectra(matrices, split=False, eigvals_only=values_only)
                split = time_spectra(matrices, split=True, eigvals_only=values_only)
                timings[mode] = {"unsplit": unsplit, "split": split,
                    "speedup": unsplit["median_seconds"]/split["median_seconds"]}
            frozen.append({"n": n, "p_total": p, "input_cutoff": cutoff,
                "payload": str(payload), "snapshot_source_path": str(source_path),
                "payload_sha256": payload_hash,
                "coefficient_sha256": hashlib.sha256(np.asarray(psi,dtype='<c16').tobytes()).hexdigest(),
                "num_matrices": len(matrices), "ic_unsplit_bits": ic_full,
                "ic_split_bits": ic_split, "ic_difference_bits": ic_split-ic_full,
                "maximum_eigenvalue_relative_error": max(c["full_eigenvalues_relative_error"] for c in case_records),
                "maximum_log_entry_difference": max(c["floored_log_maximum_absolute_entry_difference"] for c in case_records),
                "maximum_log_relative_frobenius_difference": max(c["floored_log_relative_frobenius_difference"] for c in case_records),
                "passed_spectral_checks": all(c["passed_spectral_checks"] for c in case_records),
                "timings": timings, "matrices": case_records})
            print(json.dumps({k:v for k,v in frozen[-1].items() if k not in ["matrices"]}), flush=True)

    report = {"seed": 20261011, "blas_threads": 1,
        "helper_source_sha256": hashlib.sha256((ROOT/"spectral_blocks.py").read_bytes()).hexdigest(),
        "no_tolerance_structural_checks_passed": True,
        "small_matrix_checks": records, "frozen_witness_checks": frozen,
        "passed_spectral_checks": all(r["passed_spectral_checks"] for r in records) and all(r["passed_spectral_checks"] for r in frozen),
        "elapsed_seconds": time.perf_counter()-start,
        "evidence_status": "floating_point_equivalence_checks_and_local_timing_not_a_certificate"}
    output = ROOT/"baseline/spectral_blocks_benchmark.json"
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"report":str(output), "passed":report["passed_spectral_checks"],
                      "elapsed_seconds":report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
