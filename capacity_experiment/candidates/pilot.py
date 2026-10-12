"""Sequential, all-sector screening of explicitly sourced candidate seeds.

Run from any working directory:
  OPENBLAS_NUM_THREADS=1 python candidates/pilot.py

Every reported coherent information is floating point, not certified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import SymmetricDepolarizing
from candidates.families import initial_candidates


def run(ns=(24, 45, 60), p_total=0.194868, *, per_size_seconds=180.0):
    results = []
    skipped = []
    summaries = []
    output = HERE / "pilot_results.json"
    with threadpool_limits(limits=1):
        for n in ns:
            start = time.perf_counter()
            # All input levels and all spin sectors are retained in the model.
            model = SymmetricDepolarizing(n, p_total, cache_dir=HERE / "cache")
            bank = initial_candidates(n, cutoffs=(9, 15, 21, n))
            bank = [c for c in bank if c.family == "gaussian_mod3" or "mod4" in c.name]
            if n < 37:
                skipped.append({"n": n, "candidate": "mele_long_mod4",
                                "reason": "Published input reaches excitation 37, above this block size; not silently truncated."})
            n_results = []
            print(json.dumps({"event": "model_ready", "n": n,
                              "p_total": p_total, "build_seconds": model.build_seconds,
                              "n_candidates": len(bank)}), flush=True)
            for candidate in bank:
                if time.perf_counter() - start > per_size_seconds:
                    skipped.append({"n": n, "candidate": candidate.name,
                                    "reason": "Per-size pilot time budget reached; remaining candidates untested."})
                    break
                then = time.perf_counter()
                value = model.evaluate(candidate.purification, gradient=False)
                elapsed = time.perf_counter() - then
                record = {**candidate.metadata(), **value, "evaluation_seconds": elapsed,
                          "all_spin_sectors": True, "numerically_certified": False}
                results.append(record)
                n_results.append(record)
                if candidate.family == "published_shape":
                    print(json.dumps({"event": "published_seed", "name": candidate.name,
                                      "ic_bits": value["ic_bits"], "seconds": elapsed}), flush=True)
            best = max(n_results, key=lambda x: x["ic_bits"])
            best_gaussian = max((r for r in n_results if r["family"] == "gaussian_mod3"),
                                key=lambda x: x["ic_bits"], default=None)
            summary = {"n": n, "p_total": p_total, "build_seconds": model.build_seconds,
                       "total_seconds": time.perf_counter() - start,
                       "evaluated": len(n_results), "best_name": best["name"],
                       "best_ic_bits": best["ic_bits"], "best_rate": best["rate"],
                       "best_gaussian": None if best_gaussian is None else
                           {k: best_gaussian[k] for k in ("name", "ic_bits", "rate", "parameters")}}
            summaries.append(summary)
            best_candidate = next(c for c in bank if c.name == best["name"])
            np.savez(HERE / f"pilot_best_n{n}.npz", purification=best_candidate.purification)
            document = {"status": "Floating-point exploratory screen; no certification or global-optimality claim.",
                        "p_convention": "total Pauli error probability; each nonidentity Pauli has probability p_total/3",
                        "model": "core.SymmetricDepolarizing; every spin sector included",
                        "summaries": summaries, "skipped": skipped, "results": results}
            output.write_text(json.dumps(document, indent=2) + "\n")
            print(json.dumps({"event": "size_complete", **summary}), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[24, 45, 60])
    parser.add_argument("--p", type=float, default=0.194868)
    parser.add_argument("--seconds", type=float, default=180.0)
    args = parser.parse_args()
    run(args.n, args.p, per_size_seconds=args.seconds)
