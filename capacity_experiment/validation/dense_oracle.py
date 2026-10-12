"""Dense oracle independent of the symmetric coefficient-extraction kernel.

Conventions
-----------
``p_per_pauli`` means D_p(X)=(1-3p)X+p(XXX+YXY+ZXZ), where the
three middle expressions denote Pauli conjugation. More unambiguously,
D_p(A)=(1-4p)A+2p*Tr(A)*I. Total nonidentity error is 3*p_per_pauli.
The full CPTP range is 0 <= p_per_pauli <= 1/3.

Qubits are ordered most significant first, and a reference register comes
before all channel inputs. Coefficient arrays have shape (reference_dim,n+1)
in the ascending-excitation Dicke basis. Their *total* squared norm is used;
rows are not separately normalized or orthogonalized.

This is floating-point validation, not a sign certificate. Negative
eigenvalues beyond a specified rounding tolerance raise an error. Entropy
does not renormalize blocks or discard positive small eigenvalues.

Primary formula references: arXiv:2608.15870v2 Appendix E; the oracle channel
itself is evaluated in the complete computational basis, without that kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log2
from typing import Mapping

import numpy as np


I2 = np.eye(2, dtype=np.complex128)
PAULIS = (
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.diag([1, -1]).astype(np.complex128),
)


def _validate_n(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 6:
        raise ValueError("Dense validation supports integer n from 1 to 6; use n<=4 routinely.")
    return int(n)


def _validate_p(p_per_pauli: float) -> float:
    p = float(p_per_pauli)
    if not np.isfinite(p) or not 0 <= p <= 1 / 3:
        raise ValueError("p_per_pauli must lie in the CPTP interval [0, 1/3].")
    return p


def _matrix(matrix: np.ndarray, *, n: int, reference_dim: int) -> np.ndarray:
    _validate_n(n)
    if not isinstance(reference_dim, (int, np.integer)) or reference_dim < 1:
        raise ValueError("reference_dim must be a positive integer.")
    a = np.asarray(matrix, dtype=np.complex128)
    size = int(reference_dim) * 2**n
    if a.shape != (size, size) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected a finite {(size, size)} matrix.")
    return a


def dicke_isometry(n: int) -> np.ndarray:
    """Columns are |D_k^n>, k=0,...,n, in the computational basis."""
    if n == 0:
        return np.ones((1, 1), dtype=np.complex128)
    _validate_n(n)
    out = np.zeros((2**n, n + 1), dtype=np.complex128)
    for x in range(2**n):
        k = x.bit_count()
        out[x, k] = 1 / np.sqrt(comb(n, k))
    return out


def schur_sectors(n: int) -> list[dict[str, int]]:
    """Exact integer dimensions, with f increasing from ceil(n/2) to n."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    result = []
    for f in range((n + 1) // 2, n + 1):
        q = n - f
        d = 2 * f - n
        multiplicity = comb(n, q) - (comb(n, q - 1) if q else 0)
        result.append({"f": f, "q": q, "d": d, "dimension": d + 1,
                       "multiplicity": multiplicity})
    return result


def singlet_dicke_embedding(n: int, f: int) -> np.ndarray:
    """One copy of U_(f,n-f): |singlet>^q tensor |D_j^(2f-n)>.

    This gives an independent way to extract each Schur block from a dense
    permutation-invariant output. It does not implement Appendix E's kernel.
    """
    _validate_n(n)
    if f not in range((n + 1) // 2, n + 1):
        raise ValueError("Invalid Schur sector f.")
    q, d = n - f, 2 * f - n
    chi = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2)
    singlets = np.ones(1, dtype=np.complex128)
    for _ in range(q):
        singlets = np.kron(singlets, chi)
    return np.kron(singlets[:, None], dicke_isometry(d))


def depolarize_pauli(matrix: np.ndarray, n: int, p_per_pauli: float,
                     reference_dim: int = 1) -> np.ndarray:
    """Apply the full channel through explicit local Pauli conjugations."""
    p = _validate_p(p_per_pauli)
    out = _matrix(matrix, n=n, reference_dim=reference_dim).copy()
    for qubit in range(n):
        current = out
        out = (1 - 3 * p) * current
        left = np.eye(reference_dim * 2**qubit, dtype=np.complex128)
        right = np.eye(2 ** (n - qubit - 1), dtype=np.complex128)
        for pauli in PAULIS:
            unitary = np.kron(np.kron(left, pauli), right)
            out = out + p * (unitary @ current @ unitary.conj().T)
    return out


def depolarize_affine(matrix: np.ndarray, n: int, p_per_pauli: float,
                      reference_dim: int = 1) -> np.ndarray:
    """Independent channel route: D_p(A)=(1-4p)A+2p Tr(A) I.

    Operates on arbitrary matrices, including off-diagonal non-Hermitian
    matrix units. The partial trace affects only the current physical qubit.
    """
    p = _validate_p(p_per_pauli)
    out = _matrix(matrix, n=n, reference_dim=reference_dim).copy()
    for qubit in range(n):
        left = reference_dim * 2**qubit
        right = 2 ** (n - qubit - 1)
        tensor = out.reshape(left, 2, right, left, 2, right)
        reduced = np.einsum("aibcid->abcd", tensor)
        replacement = (
            reduced[:, None, :, :, None, :]
            * I2[None, :, None, None, :, None]
        ).reshape(out.shape)
        out = (1 - 4 * p) * out + 2 * p * replacement
    return out


def hermitian_eigenvalues(matrix: np.ndarray, *, atol: float = 1e-11) -> np.ndarray:
    a = np.asarray(matrix, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError("Expected a finite square matrix.")
    residual = float(np.max(np.abs(a - a.conj().T), initial=0.0))
    if residual > atol:
        raise ValueError(f"Hermiticity residual {residual:.3g} exceeds tolerance {atol:.3g}.")
    return np.linalg.eigvalsh((a + a.conj().T) / 2)


def entropy_bits(matrix: np.ndarray, *, atol: float = 1e-11) -> float:
    """H(A)=-Tr A log2 A for PSD, possibly subnormalized A.

    No trace renormalization or positive-eigenvalue cutoff is performed.
    Only negative roundoff in [-atol,0] is replaced by zero, after checking.
    This operation does not certify the sign of a tiny entropy difference.
    """
    values = hermitian_eigenvalues(matrix, atol=atol)
    if values.min(initial=0.0) < -atol:
        raise ValueError(f"PSD violation: minimum eigenvalue {values.min():.3g}.")
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


def _partial_trace_reference(joint: np.ndarray, reference_dim: int, d: int) -> np.ndarray:
    return np.trace(joint.reshape(reference_dim, d, reference_dim, d), axis1=0, axis2=2)


@dataclass
class DenseResult:
    n: int
    reference_dim: int
    p_per_pauli: float
    normalization: float
    input_state: np.ndarray
    output_state: np.ndarray
    joint_output: np.ndarray
    input_entropy: float
    output_entropy: float
    joint_entropy: float
    coherent_information: float
    diagnostics: dict[str, float]

    def summary(self) -> dict:
        return {
            "n": self.n,
            "reference_dim": self.reference_dim,
            "p_per_pauli": self.p_per_pauli,
            "p_total_error": 3 * self.p_per_pauli,
            "normalization": self.normalization,
            "input_entropy_bits": self.input_entropy,
            "output_entropy_bits": self.output_entropy,
            "joint_entropy_bits": self.joint_entropy,
            "coherent_information_bits": self.coherent_information,
            "coherent_information_per_channel_use": self.coherent_information / self.n,
            "diagnostics": self.diagnostics,
            "evidence_status": "floating_point_cross_check_not_certificate",
        }


def _from_purification(amplitudes: np.ndarray, n: int, p_per_pauli: float,
                       *, method: str = "pauli") -> DenseResult:
    _validate_n(n)
    p = _validate_p(p_per_pauli)
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != 2**n:
        raise ValueError("Purification must have shape (reference_dim,2**n).")
    normalization = float(np.vdot(amplitudes, amplitudes).real)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Purification must have finite, positive total norm.")
    amplitudes = amplitudes / np.sqrt(normalization)
    reference_dim = amplitudes.shape[0]
    vector = amplitudes.reshape(-1)
    joint_input = np.outer(vector, vector.conj())
    rho = _partial_trace_reference(joint_input, reference_dim, 2**n)
    channel = {"pauli": depolarize_pauli, "affine": depolarize_affine}.get(method)
    if channel is None:
        raise ValueError("method must be 'pauli' or 'affine'.")
    joint = channel(joint_input, n, p, reference_dim)
    output = channel(rho, n, p)
    traced_output = _partial_trace_reference(joint, reference_dim, 2**n)
    joint_eigenvalues = hermitian_eigenvalues(joint)
    output_eigenvalues = hermitian_eigenvalues(output)
    input_entropy = entropy_bits(rho)
    output_entropy = entropy_bits(output)
    joint_entropy = entropy_bits(joint)
    diagnostics = {
        "input_trace_error": float(abs(np.trace(rho) - 1)),
        "output_trace_error": float(abs(np.trace(output) - 1)),
        "joint_trace_error": float(abs(np.trace(joint) - 1)),
        "partial_trace_consistency_max_abs": float(np.max(np.abs(output - traced_output))),
        "joint_hermiticity_max_abs": float(np.max(np.abs(joint - joint.conj().T))),
        "output_hermiticity_max_abs": float(np.max(np.abs(output - output.conj().T))),
        "joint_min_eigenvalue": float(joint_eigenvalues.min()),
        "output_min_eigenvalue": float(output_eigenvalues.min()),
    }
    return DenseResult(
        n, reference_dim, p, normalization, rho, output, joint,
        input_entropy, output_entropy, joint_entropy,
        output_entropy - joint_entropy, diagnostics,
    )


def dense_coherent_information_dicke(coefficients: np.ndarray, p_per_pauli: float,
                                      *, method: str = "pauli") -> DenseResult:
    """Coherent information for arbitrary-rank symmetric purification rows."""
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    if coefficients.ndim != 2 or coefficients.shape[0] == 0:
        raise ValueError("coefficients must have shape (reference_dim,n+1).")
    n = _validate_n(coefficients.shape[1] - 1)
    amplitudes = coefficients @ dicke_isometry(n).T
    return _from_purification(amplitudes, n, p_per_pauli, method=method)


def dense_coherent_information(rho: np.ndarray, n: int, p_per_pauli: float,
                                *, method: str = "pauli") -> DenseResult:
    """General small-n density input, without a symmetry restriction."""
    rho = _matrix(rho, n=n, reference_dim=1)
    if abs(np.trace(rho) - 1) > 1e-11:
        raise ValueError("rho must have unit trace.")
    values = hermitian_eigenvalues(rho)
    if values.min(initial=0.0) < -1e-11:
        raise ValueError("rho is not positive semidefinite.")
    values, vectors = np.linalg.eigh((rho + rho.conj().T) / 2)
    values = np.maximum(values, 0)
    amplitudes = (vectors * np.sqrt(values)[None, :]).T
    return _from_purification(amplitudes, n, p_per_pauli, method=method)


def dense_schur_blocks(coefficients: np.ndarray, p_per_pauli: float,
                       *, method: str = "pauli") -> dict[int, np.ndarray]:
    """Extract Appendix E's blocks by direct dense projection, n<=6."""
    result = dense_coherent_information_dicke(coefficients, p_per_pauli, method=method)
    blocks = {}
    for sector in schur_sectors(result.n):
        w = singlet_dicke_embedding(result.n, sector["f"])
        embedding = np.kron(np.eye(result.reference_dim), w)
        blocks[sector["f"]] = embedding.conj().T @ result.joint_output @ embedding
    return blocks


def audit_symmetric_blocks(coefficients: np.ndarray, p_per_pauli: float,
                           blocks: Mapping[int, np.ndarray], *, atol: float = 1e-10) -> dict:
    """Compare supplied core blocks with independent dense blocks and spectra.

    All sectors are required, even if a sector vanishes. Blocks must be
    subnormalized and reference-first. A ``passed`` result is numerical
    validation at ``atol``, not an interval certificate.
    """
    dense = dense_coherent_information_dicke(coefficients, p_per_pauli)
    sectors = schur_sectors(dense.n)
    if set(blocks) != {s["f"] for s in sectors}:
        raise ValueError("Supply every sector keyed by integer f, including zero sectors.")
    joint_spectra, output_spectra = [], []
    total_trace = 0j
    total_output_entropy = total_joint_entropy = 0.0
    block_comparison_max = 0.0
    block_results = []
    for sector in sectors:
        f, dimension, multiplicity = sector["f"], sector["dimension"], sector["multiplicity"]
        b = np.asarray(blocks[f], dtype=np.complex128)
        expected_shape = (dense.reference_dim * dimension,) * 2
        if b.shape != expected_shape:
            raise ValueError(f"Sector {f} has shape {b.shape}, expected {expected_shape}.")
        output_block = _partial_trace_reference(b, dense.reference_dim, dimension)
        eigenvalues = hermitian_eigenvalues(b, atol=atol)
        output_eigenvalues = hermitian_eigenvalues(output_block, atol=atol)
        total_trace += multiplicity * np.trace(b)
        total_joint_entropy += multiplicity * entropy_bits(b, atol=atol)
        total_output_entropy += multiplicity * entropy_bits(output_block, atol=atol)
        joint_spectra.append(np.repeat(eigenvalues, multiplicity))
        output_spectra.append(np.repeat(output_eigenvalues, multiplicity))
        embedding = np.kron(np.eye(dense.reference_dim), singlet_dicke_embedding(dense.n, f))
        projected = embedding.conj().T @ dense.joint_output @ embedding
        comparison = float(np.max(np.abs(b - projected)))
        block_comparison_max = max(block_comparison_max, comparison)
        block_results.append({
            **sector,
            "trace": float(np.trace(b).real),
            "sector_probability": float((multiplicity * np.trace(b)).real),
            "joint_min_eigenvalue": float(eigenvalues.min()),
            "output_min_eigenvalue": float(output_eigenvalues.min()),
            "dense_entry_error_max_abs": comparison,
        })
    joint_spectrum = np.sort(np.concatenate(joint_spectra))
    output_spectrum = np.sort(np.concatenate(output_spectra))
    block_ic = total_output_entropy - total_joint_entropy
    errors = {
        "weighted_trace_error": float(abs(total_trace - 1)),
        "joint_spectrum_max_abs": float(np.max(np.abs(joint_spectrum - np.linalg.eigvalsh(dense.joint_output)))),
        "output_spectrum_max_abs": float(np.max(np.abs(output_spectrum - np.linalg.eigvalsh(dense.output_state)))),
        "joint_entropy_abs": abs(total_joint_entropy - dense.joint_entropy),
        "output_entropy_abs": abs(total_output_entropy - dense.output_entropy),
        "coherent_information_abs": abs(block_ic - dense.coherent_information),
        "block_entry_max_abs": block_comparison_max,
    }
    return {
        "n": dense.n, "reference_dim": dense.reference_dim,
        "p_per_pauli": dense.p_per_pauli, "p_total_error": 3 * dense.p_per_pauli,
        "dense_coherent_information_bits": dense.coherent_information,
        "block_coherent_information_bits": block_ic,
        "tolerance": atol,
        "errors": errors,
        "sectors": block_results,
        "passed": all(error <= atol for error in errors.values()),
        "evidence_status": "floating_point_cross_check_not_certificate",
    }


def maximally_mixed_one_qubit_formula(p_per_pauli: float) -> float:
    p = _validate_p(p_per_pauli)
    probabilities = (1 - 3 * p, p, p, p)
    return 1 + sum(x * log2(x) for x in probabilities if x > 0)
