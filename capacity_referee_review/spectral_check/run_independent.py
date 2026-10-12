"""End-to-end verification with independent integer kernel and residual spectra.

Writes only a specified new result record. The frozen experiment is read-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import platform
import resource
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from capacity_referee_review.spectral_check import residual_verifier as rv
from capacity_referee_review.kernel_derivation import independent_kernel as kernel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payload(path, n, key):
    if path.suffix == ".npz":
        with rv.np.load(path, allow_pickle=False) as data:
            array = rv.np.asarray(data[key], dtype=rv.np.complex128)
    else:
        raw = rv.np.fromfile(path, dtype="<c16")
        if raw.size % (n+1):
            raise ValueError("Payload size does not match n.")
        array = raw.reshape(-1, n+1)
    if array.ndim != 2 or array.shape[1] != n+1 or array.shape[0] == 0:
        raise ValueError("Wrong coefficient-array shape.")
    if not rv.np.isfinite(array).all():
        raise ValueError("Nonfinite input coefficients.")
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p-total", required=True)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--precision-bits", type=int, default=512)
    parser.add_argument("--key", default="purification")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-record", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing result record.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    coefficients = load_payload(args.payload, args.n, args.key)
    norm = sum((Fraction(float(z.real))**2 + Fraction(float(z.imag))**2
                for z in coefficients.flat), Fraction(0))
    if norm <= 0:
        raise ValueError("Zero input state.")
    input_hash = sha(args.payload)
    sources = [Path(__file__), Path(rv.__file__), Path(kernel.__file__)]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    started = time.perf_counter()
    start_cpu = resource.getrusage(resource.RUSAGE_SELF)
    started_utc = datetime.now(timezone.utc).isoformat()
    rv.flint.ctx.threads = 1
    with rv.flint.ctx.workprec(args.precision_bits):
        total_ic, total_trace = rv.arb(0), rv.acb(0)
        sectors = []
        expected_q = 0
        for q, joint, output in kernel.iter_weighted_blocks(
                args.n, args.p_total, coefficients, cutoff=args.cutoff,
                precision_bits=args.precision_bits, progress=False):
            sector_started = time.perf_counter()
            if q != expected_q:
                raise ValueError("Missing or reordered sector.")
            expected_q += 1
            size = args.n - 2*q + 1
            if joint.nrows() != coefficients.shape[0]*size or joint.ncols() != coefficients.shape[0]*size:
                raise ValueError("Wrong joint-block dimension.")
            if output.nrows() != size or output.ncols() != size:
                raise ValueError("Wrong output-block dimension.")
            joint_trace = sum((joint[i, i] for i in range(joint.nrows())), rv.acb(0))
            weight = sum((output[i, i] for i in range(size)), rv.acb(0))
            if not (joint_trace - weight).contains(0):
                raise ValueError("Joint and output trace enclosures disagree.")
            total_trace += weight
            h_joint, joint_diagnostics = rv.residual_entropy(joint)
            h_output, output_diagnostics = rv.residual_entropy(output)
            contribution = h_output - h_joint
            total_ic += contribution
            sectors.append({
                "q": q,
                "dimension": size,
                "multiplicity": comb(args.n, q) - (comb(args.n, q-1) if q else 0),
                "weighted_sector_probability": rv.record(weight.real),
                "coherent_information_bits": rv.record(contribution),
                "joint_spectrum": joint_diagnostics,
                "output_spectrum": output_diagnostics,
                "spectral_seconds": time.perf_counter() - sector_started,
            })
            print(json.dumps({"q": q, "dimension": size,
                              "ic": contribution.str(18, more=True),
                              "cumulative_ic": total_ic.str(18, more=True),
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)
        if expected_q != args.n//2+1:
            raise ValueError("Incomplete sector enumeration.")
        if not total_trace.real.contains(1) or not total_trace.imag.contains(0):
            raise ValueError("Total trace enclosure does not contain one.")
        if not total_ic.is_finite():
            raise ValueError("Nonfinite coherent-information enclosure.")
        result = {
            "status": "completed",
            "n": args.n,
            "rank_rows": coefficients.shape[0],
            "input_cutoff": args.cutoff if args.cutoff is not None else args.n,
            "p_total_exact": str(Fraction(args.p_total)),
            "p_per_pauli_exact": str(Fraction(args.p_total)/3),
            "coefficient_semantics": "exact IEEE754 binary64 dyadics, exact total squared norm normalization",
            "exact_squared_norm": str(norm),
            "payload_path": str(args.payload.resolve()),
            "payload_file_sha256": input_hash,
            "full_coefficient_bytes_sha256": hashlib.sha256(rv.np.asarray(coefficients,dtype='<c16').tobytes(order='C')).hexdigest(),
            "source_hashes": source_hashes,
            "kernel": "independent exact-integer generating-column kernel; no frozen-evaluator import",
            "spectral_method": "approximate eigenpair centers; rigorous Gram/reconstruction residual bounds; polar decomposition and ordered Weyl intervals",
            "certified_acb_eig_called": False,
            "physical_spectral_constraint": "exact weighted channel blocks are Hermitian positive semidefinite with trace at most one, by the kernel derivation",
            "precision_bits": args.precision_bits,
            "python": sys.version,
            "python_executable": sys.executable,
            "python_flint_version": rv.flint.__version__,
            "numpy_version": rv.np.__version__,
            "flint_module_path": rv.flint.__file__,
            "platform": platform.platform(),
            "flint_threads": rv.flint.ctx.threads,
            "coherent_information_bits": rv.record(total_ic),
            "rate_qubits_per_channel_use": rv.record(total_ic/args.n),
            "weighted_total_trace": rv.record(total_trace.real),
            "sign": "positive" if total_ic > 0 else "negative" if total_ic < 0 else "unresolved",
            "all_spin_sectors_included": True,
            "sectors": sectors,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter()-started,
            "scope": "internal independent-method cross-check, not external replication; FLINT/Arb arithmetic and exact channel derivation remain trusted",
        }
    end_cpu = resource.getrusage(resource.RUSAGE_SELF)
    result["cpu_user_seconds"] = end_cpu.ru_utime-start_cpu.ru_utime
    result["cpu_system_seconds"] = end_cpu.ru_stime-start_cpu.ru_stime
    result["process_max_rss_kib"] = end_cpu.ru_maxrss
    result["sources_unchanged_during_run"] = source_hashes == {str(p.relative_to(ROOT)): sha(p) for p in sources}
    result["input_unchanged_during_run"] = input_hash == sha(args.payload)
    if not result["sources_unchanged_during_run"] or not result["input_unchanged_during_run"]:
        raise RuntimeError("Inputs or source changed during execution.")
    if args.compare_record:
        old = json.loads(args.compare_record.read_text())
        a, b = result["coherent_information_bits"], old["coherent_information_bits"]
        alo, ahi = Fraction(a["lower_rational"]), Fraction(a["upper_rational"])
        blo, bhi = Fraction(b["lower_rational"]), Fraction(b["upper_rational"])
        result["comparison"] = {
            "record_path": str(args.compare_record.resolve()),
            "record_sha256": sha(args.compare_record),
            "intervals_overlap": alo <= bhi and blo <= ahi,
            "new_contains_original": alo <= blo and bhi <= ahi,
            "same_sign": result["sign"] == old["sign"],
        }
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sectors"},indent=2), flush=True)


if __name__ == "__main__":
    main()
