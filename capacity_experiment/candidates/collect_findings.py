"""Collect the completed search branch into concise, source-linked notes."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def collect():
    pilot = read("pilot_results.json")
    first = read("optimization_summary.json")
    negative = read("optimization_summary_deep1000.json")
    gradual = read("optimization_summary_gradual_mem40.json")
    selected = next(s for s in gradual["stages"] if s["p_total"] == .196)
    coarse = read("cutoff_ablation_n60_p01955.json")
    edge = read("cutoff_ablation_n60_p01955_edge.json")
    complete = read("cutoff_complete_lower_scan.json")
    ablations = sorted(coarse["ablations"] + edge["ablations"],
                       key=lambda r:r["truncated_input_cutoff"])
    certificates = []
    for path in sorted((HERE.parent / "validation").glob("*.json")):
        try:
            doc=json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if doc.get("payload_file_sha256") == selected["raw_sha256"]:
            certificates.append({"path":str(path),
                                 "evidence_status":doc.get("evidence_status"),
                                 "p_total_exact":doc.get("p_total_exact"),
                                 "ic_bits":doc.get("coherent_information_bits"),
                                 "rate":doc.get("achievable_rate_lower_bound_bits_per_channel_use"),
                                 "sign":doc.get("sign"),
                                 "precision_bits":doc.get("precision_bits"),
                                 "all_spin_sectors_included":doc.get("all_spin_sectors_included")})
    inventory = ["pilot_results.json", "optimization_summary.json",
                 "refine_seed_comparison.json", "p196_interpolation_screen.json",
                 "optimization_summary_deep1000.json",
                 "optimization_summary_gradual_mem40.json",
                 "cutoff_ablation_n60_p01955.json", "cutoff_ablation_n60_p01955_edge.json",
                 "cutoff_complete_lower_scan.json", "residue_offset_screen.json"]
    document = {
        "scope": "Finite-block rank-two real modulo-three branch; no global-optimality or novelty claim.",
        "p_convention": "Total Pauli error; each of X,Y,Z has probability p/3.",
        "selected_n60_p0196_candidate": selected,
        "selected_candidate_certificates": certificates,
        "pilot_summaries": pilot["summaries"],
        "initial_continuation": first["stages"],
        "negative_basin": negative["stages"],
        "gradual_continuation": gradual["stages"],
        "cutoff_ablation": ablations,
        "minimum_positive_hard_cutoff_of_frozen_p01955_state": complete["minimum_positive_cutoff_in_scan"],
        "record_inventory": inventory,
        "source_urls": ["https://arxiv.org/html/2609.39747v2",
                        "https://arxiv.org/html/2607.27449v1"],
        "optimization_stopped_after_selected_stage": True,
    }
    (HERE / "branch_summary.json").write_text(json.dumps(document,indent=2)+"\n")

    lines = [
        "# Candidate branch: final findings",
        "",
        f"**Selected result:** a real rank-two modulo-three input for 60 channel uses gives floating-point coherent information **{selected['ic_bits']:.15g} bits** at total Pauli error **p=0.196**, corresponding to a quantum communication rate lower bound of **{selected['rate']:.15g} qubits/use**. Both mixture weights remain nonzero: {selected['row_weights'][0]:.9f} and {selected['row_weights'][1]:.9f}. All 31 output spin sectors are included.",
        "",
        f"The frozen raw payload is `{Path(selected['final_raw_c128']).name}`, SHA256 `{selected['raw_sha256']}`. It stores little-endian complex128 coefficients with shape `(2,61)` in normalized Dicke basis. Exact numerical certification must normalize the exact binary64 coefficient values as a whole.",
        "",
    ]
    if certificates:
        for certificate in certificates:
            lines.append(f"The independent certificate `{Path(certificate['path']).name}` reports `{certificate['ic_bits']['ball']}` bits at exact p={certificate['p_total_exact']}, using {certificate['precision_bits']} bits of ball arithmetic. It establishes the sign for this fixed witness, trusting the exact channel implementation and FLINT/Arb.")
    else:
        lines.append("The selected payload has been sent to the independent verification branch; its rigorous certificate was not yet present when these notes were generated. Earlier p=0.1955 and cutoff-55 states already have separate positive certificates.")
    lines += [
        "",
        "## Provenance and construction",
        "",
        "The structural prior is the rank-two real comb in [Ahmed et al., arXiv:2609.39747v2, Eq. (20) and nearby discussion](https://arxiv.org/html/2609.39747v2): one codeword uses Fock labels 0 mod 3, the other 1 mod 3. The paper describes a cutoff-220 optimized Gaussian-channel witness, but the inspected full text does not list its full amplitudes. Our finite-Dicke initializer is independently specified and does not reproduce that unpublished coefficient list.",
        "",
        "Our successful initializer has n=60, input cutoff 60, equal row weights, and amplitudes proportional to exp(-(k−24)^2/(4·9²)) separately normalized on residues 0 and 1 modulo 3. We then freed all 41 real supported amplitudes and normalized the entire purification. The row weights were free during optimization. No quantum hardware was used.",
        "",
        "Reproducible source controls came from [Mele et al., arXiv:2607.27449v1](https://arxiv.org/html/2607.27449v1): the exact cutoff-9 coefficients in Eqs. (10)–(11), and the exact cutoff-37 integer sequences in Appendix B, Eqs. (31)–(33). Their bosonic-channel certifications do not transfer to the finite-qubit Dicke lifts. The code also defines explicitly labeled modulo-three support compressions and optional rank-three modulo-five controls; these were not optimized in the completed successful branch. Rank-three modulo-five families already appear in the Mele paper.",
        "",
        "## Initial fixed-noise screen",
        "",
        "At p=0.194868, every output spin sector was retained. Values are Ic in bits:",
        "",
        "| n | Exact Mele cutoff-9 lift | Exact Mele cutoff-37 lift | Best heuristic Gaussian modulo 3 |",
        "|---:|---:|---:|---:|",
    ]
    for n in (24,45,60):
        entries=[r for r in pilot["results"] if r["n"]==n]
        short=next(r["ic_bits"] for r in entries if r["name"]==f"mele_short_mod4_M{n}")
        long=next((f"{r['ic_bits']:.10g}" for r in entries if r["name"]==f"mele_long_mod4_M{n}"),"Inapplicable:37>n")
        gauss=max(r["ic_bits"] for r in entries if r["family"]=="gaussian_mod3")
        lines.append(f"| {n} | {short:.10g} | {long} | {gauss:.10g} |")
    lines += [
        "",
        "The screen evaluated 73, 74, and 74 states at n=24, 45, and 60, taking about 1.96, 4.18, and 7.18 seconds respectively on this runtime with one BLAS thread. The different cutoffs and parameter counts preclude interpreting this table as an isolated causal comparison of modulo spacing.",
        "",
        "## Local search and the controlled basin comparison",
        "",
        "L-BFGS-B maximized 1e4·Ic using the analytic coefficient gradient, ftol=1e−13, gtol=1e−9, and at most 40 line-search steps. Initial stages used 250 iterations and 10 correction vectors. The gradual branch used up to 1000 iterations and 40 correction vectors for 41 real variables.",
        "",
        "| Route / stage | p | Best completed Ic bits | Iterations | Optimizer stopping status |",
        "|---|---:|---:|---:|---|",
    ]
    for route,stages in (("Initial continuation",first["stages"]),("Other basin",negative["stages"]),("Gradual continuation",gradual["stages"])):
        for s in stages:
            lines.append(f"| {route} | {s['p_total']} | {s['ic_bits']:.12g} | {s['optimizer_iterations']} | {s['optimizer_message']} |")
    lines += [
        "",
        "At p=0.196, starting from the p=0.197 negative branch remained negative after 1000 iterations, while gradual continuation from the p=0.1955 positive basin reached a positive state. A nine-point amplitude interpolation also showed a barrier between those seeds. This is a useful local-search observation, not a general theorem about the landscape. A stopping status or an iteration limit is not a proof of optimality. No stage counted a pure-state collapse to zero as a positive result.",
        "",
        "## Cutoff ablation of the frozen p=0.1955 state",
        "",
        "These are hard truncations followed by global renormalization, without reoptimizing coefficients. Output sectors are all retained.",
        "",
        "| Cutoff K | Original tail probability removed | Ic bits |",
        "|---:|---:|---:|",
    ]
    for r in ablations:
        lines.append(f"| {r['truncated_input_cutoff']} | {r['tail_probability_mass']:.12g} | {r['ic_bits']:.12g} |")
    lines += [
        "",
        "A complete scan of all 37 distinct nonpure hard truncations through K=55 found 55 to be the first positive cutoff for this frozen state. All smaller cutoffs were negative in floating point; the largest such value was −1.3650e−5 at K=52. Cutoff 0 is pure and has exactly zero Ic. The K=55 candidate was independently certified positive: Ic=1.38677338054150817069616050722749694e−6 ±4.60e−52. Its certificate is `../validation/ablation_n60_K55_arb_p0p1955.json`.",
        "",
        "This ablation shows that a small tail can matter for a small entropy difference. It does not yet produce a witness with K much smaller than n, and it does not establish a minimum cutoff after reoptimization. An additional six-evaluation excitation-offset control (shifts −1, 0, and +1 at p=0.1955 and 0.196) found no improvement from either shifted state.",
        "",
        "## Scope and records",
        "",
        "These results concern explicit finite-block inputs. They do not claim a new overall depolarizing-capacity record: the September paper already gives a stronger asymptotic lower bound. No literature-wide novelty claim follows from this limited search. Negative candidates are not capacity converses; positive certified Ic gives an achievable lower bound for that exact channel and witness. Comparisons across block sizes must use Ic/n as well as Ic.",
        "",
        "Optimization stopped after the selected n=60, p=0.196 stage. No further continuation to 0.1965 or 0.197 was started. `branch_summary.json` collects the structured results. `DESIGN.md` records the detailed family definitions, published integer seeds, conventions, search choices, and certification scope. The full scan records are:",
        "",
    ]
    lines.extend(f"- `{name}`" for name in inventory)
    lines += ["", "Code: `families.py`, `pilot.py`, `optimize_real.py`, `cutoff_ablation.py`, and this collector. Frozen `*_final.npz` / `.c128` payloads are protected against accidental overwrite; repeated optimization uses a new `--label`.", ""]
    (HERE / "FINDINGS.md").write_text("\n".join(lines))
    print(json.dumps({"findings":str(HERE/"FINDINGS.md"),"summary":str(HERE/"branch_summary.json"),"selected_ic_bits":selected["ic_bits"],"certificates":len(certificates)},indent=2))


if __name__ == "__main__":
    collect()
