# Finite-block quantum capacity: experiment code

Source code by **Ahmet Kamer Çivi**, Middle East Technical University, Physics Department, for *Certified finite block lower bounds for the qubit depolarizing channel*.

This release contains the finite-channel evaluator, numerical optimization, candidate generation, interval verification, small-system checks, and a second internal channel/spectrum implementation. Frozen input datasets and recorded computational results are not included in this code release.

The public noise convention is total nonidentity Pauli probability:

`D_p(rho) = (1-p) rho + (p/3) (X rho X + Y rho Y + Z rho Z)`.

## Quick start

Use Python 3.12. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python capacity_experiment/core.py --n 3 --p 0.19
.venv/bin/python capacity_experiment/validation/run_core_checks.py \
  --output replays/core_checks.json
```

The small-system checks compare the reduced channel with a full computational-basis implementation and independently check the gradient.

## Generate and optimize new candidates

```sh
.venv/bin/python capacity_experiment/candidates/pilot.py \
  --n 24 --p 0.194868 --seconds 60
.venv/bin/python capacity_experiment/candidates/optimize_real.py \
  --seed capacity_experiment/candidates/pilot_best_n24.npz \
  --p 0.194868 --maxiter 100 --label example
```

These are exploratory floating-point searches. A positive numerical result needs a separate interval calculation before it supports a certified claim.

## Verify a supplied input

For a local `input.c128` containing little-endian complex128 coefficients with shape `(reference_rank, n+1)`:

```sh
.venv/bin/python capacity_experiment/validation/interval_core.py \
  --payload input.c128 --n 45 --p-total 1949/10000 \
  --precision-bits 192 --output replays/input_interval.json
```

For larger sparse inputs use `validation/interval_split.py`, an exact-zero component splitter, and sufficient precision. `--p-total` accepts exact rational values. Normalize the total purification, not its rows separately. Every output spin sector is retained.

The second internal implementation is in `capacity_referee_review/kernel_derivation/independent_kernel.py` and `capacity_referee_review/spectral_check/residual_verifier.py`. It can be invoked with `capacity_referee_review/spectral_check/run_independent.py` and a supplied payload. Both interval methods depend on FLINT/Arb. They are internal computational checks, not external peer review.

Analysis scripts generate figures from locally available experiment data. Scripts that inventory saved certificates or collect historical findings require those records to be supplied separately.

## Sources

The channel formula follows Appendix E of [arXiv:2608.15870v2](https://arxiv.org/abs/2608.15870v2). Gaussian candidate families are explicit heuristic choices inspired by [arXiv:2609.39747v2](https://arxiv.org/abs/2609.39747v2). The unchanged published integer verifier is included under `capacity_experiment/baseline/zenodo_v1_0/artifact/verifier/`; its certificate is available from the [original public deposit](https://doi.org/10.5281/zenodo.21962925).

The work concerns explicit finite inputs and their evaluations. It does not claim a new global capacity threshold, optimality, or a complete practical decoder.

## AI assistance and attribution

ChatGPT assistance was used in writing the code, preparing the manuscript, and performing internal computational checks. The human author is responsible for the scientific claims and released material.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the published verifier and candidate-family attribution. `SHA256SUMS` records the released source-file fingerprints.
