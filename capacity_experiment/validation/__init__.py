"""Independent, small-n validation for depolarizing coherent information."""

from .dense_oracle import (
    DenseResult,
    audit_symmetric_blocks,
    dense_coherent_information,
    dense_coherent_information_dicke,
    dense_schur_blocks,
    depolarize_affine,
    depolarize_pauli,
    dicke_isometry,
    entropy_bits,
    schur_sectors,
    singlet_dicke_embedding,
)

__all__ = [
    "DenseResult", "audit_symmetric_blocks", "dense_coherent_information",
    "dense_coherent_information_dicke", "dense_schur_blocks",
    "depolarize_affine", "depolarize_pauli", "dicke_isometry",
    "entropy_bits", "schur_sectors", "singlet_dicke_embedding",
]
