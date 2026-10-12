"""Local continuation of a real comb purification with free row weights.

Each real amplitude on the declared support is optimized directly. The core
normalizes the entire purification; no separate codeword normalization is
performed during this optimization. All coherent-information values here are
floating-point estimates, not certificates. Stable final payloads are distinct
from the overwritten, in-progress best checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import SymmetricDepolarizing


def _tag(p):
    return f"{p:.6f}".replace(".", "p")


def optimize(seed_path, noises=(.194868, .1955, .197), maxiter=250, label="", maxcor=10):
    seed_path = Path(seed_path)
    z = np.load(seed_path, allow_pickle=False)
    C = np.asarray(z["purification"], dtype=complex)
    if C.ndim != 2 or np.max(np.abs(C.imag)) > 1e-13:
        raise ValueError("A real rank-by-Dicke purification is required.")
    C /= np.linalg.norm(C)
    n = C.shape[1] - 1
    support = np.nonzero(C.real)
    x = C.real[support].copy()
    summaries = []
    started = time.perf_counter()
    if label and (not label.replace("_", "").replace("-", "").isalnum()):
        raise ValueError("Run label must use only letters, digits, underscores, and hyphens.")
    stage_seed_path = seed_path
    with threadpool_limits(limits=1):
        for p in noises:
            suffix = f"_{label}" if label else ""
            prefix = HERE / f"gaussian_n{n}_p{_tag(p)}{suffix}"
            finalfile = prefix.with_name(prefix.name + "_final.npz")
            if finalfile.exists():
                raise FileExistsError(f"Refusing to overwrite frozen payload {finalfile}; select a new --label.")
            logfile = prefix.with_name(prefix.name + "_log.jsonl")
            log = logfile.open("w")
            model = SymmetricDepolarizing(n, p, cache_dir=HERE / "cache")
            stage_start = time.perf_counter()
            best = {"ic_bits": -np.inf, "x": None, "evaluation": None}
            counters = {"evaluations": 0, "iterations": 0}

            def unpack(v):
                psi = np.zeros_like(C)
                psi[support] = v
                return psi

            def emit(record):
                log.write(json.dumps(record) + "\n")
                log.flush()

            def objective(v):
                counters["evaluations"] += 1
                psi = unpack(v)
                value = model.evaluate(psi, gradient=True)
                gradient = value.pop("gradient").real[support]
                if not np.isfinite(value["ic_bits"]) or not np.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite objective or gradient")
                improved = value["ic_bits"] > best["ic_bits"]
                if improved:
                    best.update(ic_bits=value["ic_bits"], x=v.copy(), evaluation=value.copy())
                    normalized = psi / np.linalg.norm(psi)
                    np.savez(prefix.with_name(prefix.name + "_best.npz"),
                             purification=normalized, p_total=p,
                             ic_bits=value["ic_bits"], numerically_certified=False)
                emit({"event": "evaluation", **counters, **value,
                      "gradient_norm": float(np.linalg.norm(gradient)),
                      "row_weights": (np.sum(np.abs(psi)**2, axis=1) / np.vdot(psi, psi).real).tolist(),
                      "best_ic_bits": best["ic_bits"], "new_best": improved,
                      "seconds": time.perf_counter() - stage_start})
                return -1e4 * value["ic_bits"], -1e4 * gradient

            def callback(v):
                counters["iterations"] += 1
                if counters["iterations"] % 25 == 0:
                    print(json.dumps({"event": "progress", "n": n, "p_total": p,
                                      **counters, "best_ic_bits": best["ic_bits"],
                                      "seconds": time.perf_counter() - stage_start}), flush=True)

            initial = model.evaluate(unpack(x))
            print(json.dumps({"event": "stage_start", "n": n, "p_total": p,
                              "initial_ic_bits": initial["ic_bits"],
                              "real_variables": len(x), "all_spin_sectors": True,
                              "build_seconds": model.build_seconds}), flush=True)
            result = minimize(objective, x, method="L-BFGS-B", jac=True,
                              callback=callback,
                              options={"ftol": 1e-13, "gtol": 1e-9,
                                       "maxiter": maxiter, "maxls": 40, "maxcor": maxcor})
            x = best["x"]
            Cbest = unpack(x)
            Cbest /= np.linalg.norm(Cbest)
            final = model.evaluate(Cbest, diagnostics=True)
            np.savez(finalfile, purification=Cbest, p_total=p,
                     ic_bits=final["ic_bits"], numerically_certified=False)
            rawfile = prefix.with_name(prefix.name + "_final.c128")
            Cbest.astype("<c16").tofile(rawfile)
            summary = {"status": "Uncertified floating-point local optimization",
                       "n": n, "p_total": p, "all_spin_sectors": True,
                       "source_seed": str(stage_seed_path),
                       "original_seed": str(seed_path), "run_label": label,
                       "family": "rank-two real mod-three comb, residues zero and one",
                       "real_variables": len(x), "free_row_weights": True,
                       "initial_ic_bits": initial["ic_bits"], **final,
                       "row_weights": np.sum(np.abs(Cbest)**2, axis=1).tolist(),
                       "elapsed_seconds": time.perf_counter() - stage_start,
                       **counters, "optimizer_success": bool(result.success),
                       "optimizer_message": str(result.message),
                       "optimizer_iterations": int(result.nit),
                       "optimizer_settings": {"method": "L-BFGS-B", "scale": 1e4,
                                              "ftol": 1e-13, "gtol": 1e-9,
                                              "maxiter": maxiter, "maxls": 40,
                                              "maxcor": maxcor},
                       "final_npz": str(finalfile), "final_raw_c128": str(rawfile),
                       "raw_sha256": hashlib.sha256(rawfile.read_bytes()).hexdigest()}
            prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            summaries.append(summary)
            emit({"event": "stage_complete", **summary})
            log.close()
            print(json.dumps({k: v for k, v in summary.items() if k != "sectors"}), flush=True)
            (HERE / f"optimization_summary{suffix}.json").write_text(json.dumps({
                "status": "Floating-point exploration, not numerical certification or global optimization.",
                "elapsed_seconds": time.perf_counter() - started, "stages": summaries}, indent=2) + "\n")
            stage_seed_path = finalfile
            if final["ic_bits"] <= 0:
                print(json.dumps({"event": "continuation_stopped", "reason": "Best local value is nonpositive."}), flush=True)
                break
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=str(HERE / "pilot_best_n60.npz"))
    parser.add_argument("--p", type=float, nargs="+", default=[.194868, .1955, .197])
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--label", default="", help="Unique suffix for a new run; frozen payloads are never overwritten.")
    parser.add_argument("--maxcor", type=int, default=10, help="L-BFGS correction history size.")
    args = parser.parse_args()
    optimize(args.seed, args.p, args.maxiter, args.label, args.maxcor)
