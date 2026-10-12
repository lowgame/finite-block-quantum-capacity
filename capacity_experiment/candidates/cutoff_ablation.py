"""Frozen-state cutoff ablation, with global renormalization and all sectors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import SymmetricDepolarizing


def ablate(seed, p=.1955, cutoffs=(18,24,30,36,42,48,54,60), label="n60_p01955"):
    seed = Path(seed)
    C = np.load(seed, allow_pickle=False)["purification"]
    C = C / np.linalg.norm(C)
    n = C.shape[1] - 1
    rows = []
    with threadpool_limits(limits=1):
        model = SymmetricDepolarizing(n, p, cache_dir=HERE / "cache")
        baseline = model.evaluate(C)
        for K in cutoffs:
            if not 0 <= K <= n:
                raise ValueError("Cutoff must lie in the input Dicke range.")
            prefix = HERE / f"ablation_{label}_K{K}"
            rawpath = prefix.with_suffix(".c128")
            if rawpath.exists():
                raise FileExistsError(f"Frozen ablation payload exists: {rawpath}")
            tail_mass = float(np.sum(np.abs(C[:, K+1:])**2))
            state = C.copy()
            state[:, K+1:] = 0
            state /= np.linalg.norm(state)
            result = model.evaluate(state, diagnostics=True)
            np.savez(prefix.with_suffix(".npz"), purification=state,
                     p_total=p, input_cutoff=K, truncation_tail_mass=tail_mass,
                     numerically_certified=False)
            state.astype("<c16").tofile(rawpath)
            weights = np.sum(np.abs(state)**2, axis=1)
            record = {"source_seed": str(seed), "label": label,
                      "truncated_input_cutoff": K, "tail_probability_mass": tail_mass,
                      "retained_probability_mass": 1-tail_mass,
                      "global_renormalization": True, "all_output_spin_sectors": True,
                      "row_weights": weights.tolist(), **result,
                      "original_ic_bits": baseline["ic_bits"],
                      "ic_change_bits": result["ic_bits"] - baseline["ic_bits"],
                      "candidate_npz": str(prefix.with_suffix(".npz")),
                      "candidate_raw": str(rawpath),
                      "raw_sha256": hashlib.sha256(rawpath.read_bytes()).hexdigest(),
                      "status": "Uncertified floating-point fixed-state truncation; no reoptimization."}
            rows.append(record)
            print(json.dumps({k:v for k,v in record.items() if k != "sectors"}), flush=True)
    document = {"source_seed": str(seed), "n": n, "p_total": p,
                "full_input_state_ic_bits": baseline["ic_bits"], "ablations": rows,
                "minimal_positive_tested_cutoff": min((r["truncated_input_cutoff"] for r in rows if r["ic_bits"] > 0), default=None)}
    (HERE / f"cutoff_ablation_{label}.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=str(HERE / "gaussian_n60_p0p195500_final.npz"))
    parser.add_argument("--p", type=float, default=.1955)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[18,24,30,36,42,48,54,60])
    parser.add_argument("--label", default="n60_p01955")
    args=parser.parse_args()
    ablate(args.seed,args.p,args.cutoffs,args.label)
