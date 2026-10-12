"""Reproduce the frozen-state comparison data and scientific figures.

No state is optimized here. Entropies and I_c are measured in bits; I_c/n is
the coherent-information lower bound in qubits per channel use. Floating-point
zero crossings are explicitly not certified thresholds.

Run: OPENBLAS_NUM_THREADS=1 python analysis/make_analysis.py
Use --plots-only to refresh figures/certificate markers from saved data.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from core import SymmetricDepolarizing

P_GRID = [.189,.19,.192,.194,.194868,.1949,.195,.1955,.196,.1965,
          .197,.19725,.1975,.198]
STATE_SPECS = [
    {"id":"A45_original", "label":"45 uses · published A", "n":45,
     "path":"baseline/github_current/artifact/payloads/A.c128", "key":None,
     "provenance":"Published August A payload; exact source bytes verified separately.",
     "color":"#5C677D", "linestyle":"--"},
    {"id":"A45_continued", "label":"45 uses · continued A", "n":45,
     "path":"results/A45_continued.npz", "key":"psi",
     "provenance":"This session's continuation from the published A payload.",
     "color":"#0072B2", "linestyle":"-"},
    {"id":"comb60", "label":"60 uses · real comb", "n":60,
     "path":"candidates/gaussian_n60_p0p196000_gradual_mem40_final.npz", "key":"purification",
     "provenance":"This session's rank-two real modulo-three input, frozen after continuation to p=0.196.",
     "color":"#009E73", "linestyle":"-"},
    {"id":"comb120", "label":"120 uses · real comb", "n":120,
     "path":"results/n120_optimized_p0p197.npz", "key":"psi",
     "provenance":"This session's optimized n120 Gaussian-inspired input, frozen after its p=0.197 run completed.",
     "color":"#D55E00", "linestyle":"-"},
]


def write_csv(path, rows, fields):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def load_states():
    states=[]
    for spec in STATE_SPECS:
        path=BASE/spec["path"]
        if spec["key"] is None:
            C=np.fromfile(path,dtype="<c16").reshape(2,spec["n"]+1)
        else:
            C=np.array(np.load(path,allow_pickle=False)[spec["key"]],dtype=complex)
        if C.shape!=(2,spec["n"]+1) or not np.isfinite(C).all():
            raise ValueError(f"Unexpected state shape or coefficients: {path}")
        norm2=float(np.vdot(C,C).real)
        if norm2<=0:raise ValueError("Zero purification")
        nonzero=np.where(np.any(C!=0,axis=0))[0]
        cutoff=int(nonzero[-1])
        gram=C@C.conj().T/norm2
        eigenvalues=np.linalg.eigvalsh((gram+gram.conj().T)/2)
        positive=eigenvalues[eigenvalues>0]
        entropy=float(-positive@np.log2(positive))
        states.append({**spec,"C":C,"input_cutoff":cutoff,
                       "input_entropy_bits":entropy,
                       "reference_spectrum":eigenvalues.tolist(),
                       "raw_squared_norm":norm2,
                       "coefficient_sha256":hashlib.sha256(C.astype("<c16").tobytes()).hexdigest(),
                       "container_file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                       "coefficient_weights":(np.abs(C)**2/norm2).tolist()})
    return states


def measure(model,state):
    value=model.evaluate(state["C"])
    return {"state_id":state["id"],"state_label":state["label"],
            "n":state["n"],"p_total":value["p_total"],
            "ic_bits_per_block":value["ic_bits"],
            "rate_qubits_per_use":value["rate"],
            "trace":value["trace"],
            "min_eigenvalue":value["min_eigenvalue"],
            "negative_eigenvalue_mass":value["negative_eigenvalue_mass"],
            "input_cutoff":state["input_cutoff"],
            "input_entropy_bits":state["input_entropy_bits"],
            "coefficient_sha256":state["coefficient_sha256"],
            "all_spin_sectors_included":True,
            "numerically_certified":False}


def build_dataset():
    started=time.perf_counter()
    states=load_states(); rows=[]; crossings=[]
    with threadpool_limits(limits=1):
        for n in (45,60,120):
            group=[s for s in states if s["n"]==n]
            K=max(s["input_cutoff"] for s in group)
            for p in P_GRID:
                model=SymmetricDepolarizing(n,p,cutoff=K,cache_dir=BASE/"cache")
                for state in group:
                    row=measure(model,state);rows.append(row)
                del model
            print(json.dumps({"event":"grid_complete","n":n,"states":len(group),
                              "elapsed_seconds":time.perf_counter()-started}),flush=True)
        for state in states:
            grid=sorted((r for r in rows if r["state_id"]==state["id"]),key=lambda r:r["p_total"])
            pair=next(((a,b) for a,b in zip(grid,grid[1:])
                       if a["ic_bits_per_block"]>0 and b["ic_bits_per_block"]<=0),None)
            if pair is None:
                crossings.append({"state_id":state["id"],"status":"No sign change bracketed on the requested grid."})
                continue
            low,high=pair[0]["p_total"],pair[1]["p_total"]
            low_ic,high_ic=pair[0]["ic_bits_per_block"],pair[1]["ic_bits_per_block"]
            evaluations=[]
            while high-low>1e-7:
                mid=(low+high)/2
                # Intermediate kernels are intentionally not written to the large cache.
                model=SymmetricDepolarizing(state["n"],mid,cutoff=state["input_cutoff"])
                result=measure(model,state);evaluations.append(result)
                if result["ic_bits_per_block"]>0:
                    low=mid;low_ic=result["ic_bits_per_block"]
                else:
                    high=mid;high_ic=result["ic_bits_per_block"]
            crossing={"state_id":state["id"],"state_label":state["label"],
                      "n":state["n"],"numerical_zero_crossing":(low+high)/2,
                      "positive_endpoint_p":low,"negative_endpoint_p":high,
                      "positive_endpoint_ic_bits":low_ic,"negative_endpoint_ic_bits":high_ic,
                      "bracket_width_p":high-low,"bisection_tolerance_p":1e-7,
                      "coefficient_sha256":state["coefficient_sha256"],
                      "status":"Floating-point zero crossing of one frozen state; not a certified capacity threshold.",
                      "numerically_certified":False,"bisection_evaluations":evaluations}
            crossings.append(crossing)
            print(json.dumps({k:v for k,v in crossing.items() if k!="bisection_evaluations"}),flush=True)
    doc={"description":"Four frozen inputs evaluated at common total-Pauli noise values; no reoptimization.",
         "channel":"D_p(rho)=(1-p)rho+(p/3)(XrhoX+YrhoY+ZrhoZ)",
         "entropy_unit":"bits", "rate_unit":"qubits per channel use",
         "coefficient_semantics":"Original saved arrays; the channel evaluator normalizes their total squared norm. No row-wise normalization.",
         "p_grid":P_GRID,"states":[{k:v for k,v in s.items() if k!="C"} for s in states],
         "evaluations":rows,"numerical_zero_crossings":crossings,
         "all_spin_sectors_included":True,"elapsed_seconds":time.perf_counter()-started,
         "versions":{"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
         "core_source_sha256":hashlib.sha256((BASE/"core.py").read_bytes()).hexdigest()}
    (HERE/"frozen_state_comparison.json").write_text(json.dumps(doc,indent=2)+"\n")
    write_csv(HERE/"frozen_state_comparison.csv",rows,
              ["state_id","state_label","n","p_total","ic_bits_per_block","rate_qubits_per_use",
               "trace","min_eigenvalue","negative_eigenvalue_mass","input_entropy_bits",
               "input_cutoff","coefficient_sha256","all_spin_sectors_included","numerically_certified"])
    write_csv(HERE/"numerical_zero_crossings.csv",crossings,
              ["state_id","n","numerical_zero_crossing","positive_endpoint_p","negative_endpoint_p",
               "positive_endpoint_ic_bits","negative_endpoint_ic_bits","bracket_width_p",
               "bisection_tolerance_p","coefficient_sha256","status"])
    profiles=[]
    for state in states:
        if state["id"] not in ("comb60","comb120"):continue
        for a,row in enumerate(state["coefficient_weights"]):
            for k,weight in enumerate(row):
                profiles.append({"state_id":state["id"],"n":state["n"],"reference_basis_a":a,
                                 "dicke_excitation_k":k,"normalized_coefficient_weight":weight,
                                 "coefficient_sha256":state["coefficient_sha256"]})
    write_csv(HERE/"state_profiles.csv",profiles,list(profiles[0]))
    return doc


def certification_points(doc):
    points=[]
    hashes={s["coefficient_sha256"]:s for s in doc["states"]}
    for path in sorted((BASE/"validation").glob("*.json")):
        try:entry=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if not isinstance(entry,dict) or not entry.get("certified_positive_coherent_information"):continue
        digest=entry.get("payload_file_sha256",entry.get("coefficient_bytes_sha256"))
        # NPZ-backed certificates may identify the raw coefficient bytes instead.
        if digest not in hashes:digest=entry.get("coefficient_bytes_sha256")
        if digest not in hashes:continue
        state=hashes[digest]
        p=float(Fraction(entry["p_total_exact"]))
        exact=entry.get("coherent_information_bits",{})
        try:
            low=Fraction(exact["lower_rational"]);high=Fraction(exact["upper_rational"])
        except (KeyError,ValueError):continue
        center=float((low+high)/2)
        points.append({"state_id":state["id"],"p_total":p,"ic_bits_per_block":center,
                       "rate_qubits_per_use":center/state["n"],"certificate":str(path.relative_to(BASE)),
                       "coefficient_sha256":digest,"ball":exact.get("ball")})
    # Only one marker is needed if repeated certificates pin the same state and p.
    unique={(p["state_id"],p["p_total"]):p for p in points}
    return list(unique.values())


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":13,"axes.labelsize":11,
                         "figure.titlesize":16,"legend.fontsize":9,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.edgecolor":"#727A84","axes.labelcolor":"#202833",
                         "xtick.color":"#46515D","ytick.color":"#46515D",
                         "svg.fonttype":"none","savefig.facecolor":"white"})


def save(fig,name):
    for extension in ("png","svg"):
        kwargs={"dpi":300} if extension=="png" else {"metadata":{"Date":None}}
        fig.savefig(HERE/f"{name}.{extension}",bbox_inches="tight",**kwargs)
    plt.close(fig)


def make_figures(doc):
    style();certificates=certification_points(doc)
    (HERE/"certified_plot_points.json").write_text(json.dumps(certificates,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(9.5,5.6))
    positives=[]
    for state in doc["states"]:
        rows=sorted((r for r in doc["evaluations"] if r["state_id"]==state["id"]),key=lambda r:r["p_total"])
        p=np.array([r["p_total"] for r in rows]);rate=np.array([r["rate_qubits_per_use"] for r in rows])
        positives.extend(rate[rate>0]);rate[rate<=0]=np.nan
        ax.plot(p,rate,color=state["color"],linestyle=state["linestyle"],lw=2,label=state["label"])
        for point in certificates:
            if point["state_id"]==state["id"]:
                ax.scatter(point["p_total"],point["rate_qubits_per_use"],s=48,marker="D",
                           facecolor="white",edgecolor=state["color"],lw=1.5,zorder=5)
    ax.set_yscale("log");ax.set_xlim(.1888,.1982)
    ax.set_ylim(min(positives)*.4,max(positives)*1.6)
    ax.set_xlabel("Total Pauli error probability, p")
    ax.set_ylabel("Coherent-information lower bound\n(qubits per channel use)")
    ax.set_xticks([.189,.19,.192,.194,.196,.198])
    ax.grid(axis="y",which="major",alpha=.23,lw=.7)
    ax.grid(axis="y",which="minor",alpha=.07,lw=.5)
    ax.legend(loc="upper right",frameon=False)
    ax.set_title("Four frozen inputs to the depolarizing channel",loc="left",pad=14)
    fig.text(.125,.01,"Only positive I_c/n is shown. Lines connect sampled points; diamonds mark independently certified positives.\nAll output spin sectors are retained. These curves do not identify the channel's capacity.",fontsize=9,color="#59636E")
    fig.subplots_adjust(bottom=.22,top=.9)
    save(fig,"frozen_state_rates")

    coarse=json.loads((BASE/"candidates/cutoff_ablation_n60_p01955.json").read_text())["ablations"]
    edge=json.loads((BASE/"candidates/cutoff_ablation_n60_p01955_edge.json").read_text())["ablations"]
    n60=[{"n":60,"p_total":.1955,"input_cutoff":r["truncated_input_cutoff"],
          "discarded_probability":r["tail_probability_mass"],"ic_bits":r["ic_bits"],
          "source":"Frozen n60 p=.1955 state; no reoptimization"} for r in coarse+edge]
    n120=json.loads((BASE/"results/n120_seed_cutoff_ablation.json").read_text())
    ablations=sorted(n60,key=lambda r:r["input_cutoff"])+[dict(r,source="n120 Gaussian seed, not the optimized n120 profile") for r in n120]
    (HERE/"cutoff_ablation_comparison.json").write_text(json.dumps(ablations,indent=2)+"\n")
    write_csv(HERE/"cutoff_ablation_comparison.csv",ablations,
              ["n","p_total","input_cutoff","discarded_probability","ic_bits","source"])
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex="col",gridspec_kw={"height_ratios":[1.35,1]})
    for col,(n,p,title,color) in enumerate(((60,.1955,"60 uses · frozen continued state","#009E73"),
                                           (120,.197,"120 uses · Gaussian seed","#D55E00"))):
        rows=sorted((r for r in ablations if r["n"]==n),key=lambda r:r["input_cutoff"])
        x=np.array([r["input_cutoff"] for r in rows]);ic=np.array([r["ic_bits"] for r in rows]);tail=np.array([r["discarded_probability"] for r in rows])
        top,bottom=axes[:,col]
        top.plot(x,ic,color=color,lw=1.8,marker="o",ms=4)
        top.axhline(0,color="#505762",lw=.8)
        top.set_yscale("symlog",linthresh=1e-7,linscale=1)
        top.set_ylim(-.05,8e-6);top.set_ylabel("Coherent information\n(bits per block)")
        top.set_title(f"{title}\np = {p}",loc="left",pad=12)
        top.grid(axis="y",alpha=.2,lw=.6)
        if n==60:
            row=next(r for r in rows if r["input_cutoff"]==55)
            top.scatter([55],[row["ic_bits"]],marker="D",s=50,facecolor="white",edgecolor=color,lw=1.5,zorder=6)
            top.annotate("K=55: certified positive",(55,row["ic_bits"]),xytext=(26,4e-6),
                         arrowprops={"arrowstyle":"-","color":"#65717D","lw":.7},fontsize=8.5,color="#46515D")
        mask=tail>0
        bottom.plot(x[mask],tail[mask],color=color,lw=1.8,marker="o",ms=4)
        bottom.set_yscale("log");bottom.set_ylabel("Discarded input weight")
        bottom.set_xlabel("Input excitation cutoff, K");bottom.grid(axis="y",alpha=.2,lw=.6)
        zero=x[~mask]
        if len(zero):bottom.text(.04,.05,f"K={int(zero[0])}: exactly zero discarded",transform=bottom.transAxes,fontsize=8.5,color="#59636E")
        bottom.set_xticks([24,36,48,60] if n==60 else [24,36,48,60,72])
    fig.suptitle("Small input tails can determine the sign",x=.085,ha="left",y=.99)
    fig.text(.085,.012,"Hard input truncation followed by global renormalization; coefficients are not reoptimized.\nEvery output spin sector is retained. The two columns use different frozen inputs and different noise values.",fontsize=9,color="#59636E")
    fig.tight_layout(rect=(0,.08,1,.945));save(fig,"cutoff_ablation")

    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,identifier,p in zip(axes,("comb60","comb120"),(.196,.197)):
        state=next(s for s in doc["states"] if s["id"]==identifier)
        weights=np.array(state["coefficient_weights"])
        for a,(color,marker) in enumerate((("#0072B2","o"),("#D55E00","s"))):
            sites=np.where(weights[a]>0)[0]
            ax.vlines(sites,0,weights[a,sites],color=color,lw=1,alpha=.55)
            ax.scatter(sites,weights[a,sites],color=color,marker=marker,s=17,label=f"Reference basis a={a}",zorder=3)
        K=state["input_cutoff"]
        if K<state["n"]:
            ax.axvspan(K+.5,state["n"]+.5,color="#E9EDF1",alpha=.55)
            ax.text((K+state["n"])/2,weights.max()*.58,"Exactly zero\nabove K="+str(K),ha="center",fontsize=9,color="#6A7580")
        ax.set_xlim(-1,state["n"]+1);ax.set_ylim(bottom=0)
        ax.set_title(f"{state['n']} uses · frozen after p={p}",loc="left",pad=14)
        ax.set_xlabel("Dicke excitation number, k")
        ax.set_ylabel(r"Normalized coefficient weight, $|\psi_{a,k}|^2$")
        ax.grid(axis="y",alpha=.18,lw=.6);ax.legend(frameon=False,loc="upper right")
    fig.suptitle("Profiles of the selected purifications",x=.07,ha="left",y=.995)
    fig.text(.07,.012,"Weights sum to one over both reference rows and all excitations. The reference label a is not a spin-sector label.\nThese profiles describe the selected purifications; the n120 cutoff ablation uses its earlier Gaussian seed.",fontsize=9,color="#59636E")
    fig.tight_layout(rect=(0,.11,1,.93));save(fig,"state_profiles")
    return certificates


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plots-only",action="store_true")
    args=parser.parse_args()
    doc=json.loads((HERE/"frozen_state_comparison.json").read_text()) if args.plots_only else build_dataset()
    certs=make_figures(doc)
    print(json.dumps({"event":"analysis_complete","directory":str(HERE),
                      "grid_evaluations":len(doc["evaluations"]),"certificate_markers":len(certs)},indent=2),flush=True)
