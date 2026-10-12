"""Floating-point Hermitian eigensolves using exact matrix-zero structure.

The graph retains every nonzero entry, regardless of magnitude, in either
triangle. Its connected components therefore describe a permutation direct
sum of the supplied floating-point matrix. No tolerance is applied. This is
an acceleration of ordinary SciPy eigh, not an interval certificate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as _scipy_eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def exact_zero_components(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return sorted coordinate arrays for exact-zero graph components.

    Both triangles contribute edges. NaNs also compare nonzero and thus
    cannot accidentally create a structural zero; finiteness is checked by
    the eigensolver when requested. Diagonal entries never connect vertices.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A square two-dimensional matrix is required.")
    if not matrix.shape[0]:
        return ()
    edges = matrix != 0
    edges |= edges.T
    np.fill_diagonal(edges, False)
    count, labels = connected_components(csr_matrix(edges), directed=False)
    return tuple(np.flatnonzero(labels == component) for component in range(count))


def eigh_exact_blocks(matrix: np.ndarray, *, eigvals_only: bool = False,
                      lower: bool = True, check_finite: bool = True,
                      driver: str | None = None,
                      overwrite_a: bool = False):
    """Compute all eigenvalues, and optionally full-coordinate eigenvectors.

    Return values follow scipy.linalg.eigh: sorted eigenvalues, or their
    pair with an eigenvector matrix whose columns are the corresponding
    eigenvectors. A disconnected component's vectors are embedded into the
    original matrix coordinates before globally sorting the spectrum.

    The Hermitian-input contract is the same as SciPy's. Components are
    detected from both triangles; ``lower`` chooses the triangle passed to
    each SciPy solve. Generalized and subset eigensolves are not supported.
    ``overwrite_a`` is honored only on the unsplit fast path; extracted
    component arrays are independent copies.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A square two-dimensional matrix is required.")
    if check_finite and not np.all(np.isfinite(matrix)):
        raise ValueError("The input matrix contains an infinity or NaN.")
    components = exact_zero_components(matrix)
    size = len(matrix)
    if len(components) <= 1:
        return _scipy_eigh(matrix, eigvals_only=eigvals_only, lower=lower,
                           check_finite=False, driver=driver,
                           overwrite_a=overwrite_a)

    eigenvalues = np.empty(size, dtype=float)
    eigenvectors = None if eigvals_only else np.zeros(
        (size, size), dtype=np.result_type(matrix.dtype, np.float64))
    column = 0
    for coordinates in components:
        width = len(coordinates)
        if width == 1:
            eigenvalues[column] = np.real(matrix[coordinates[0], coordinates[0]])
            if eigenvectors is not None:
                eigenvectors[coordinates[0], column] = 1
        else:
            block = matrix[np.ix_(coordinates, coordinates)]
            result = _scipy_eigh(block, eigvals_only=eigvals_only, lower=lower,
                                 check_finite=False, driver=driver,
                                 overwrite_a=True)
            if eigvals_only:
                eigenvalues[column:column+width] = result
            else:
                values, vectors = result
                eigenvalues[column:column+width] = values
                eigenvectors[np.ix_(coordinates, np.arange(column, column+width))] = vectors
        column += width
    order = np.argsort(eigenvalues, kind="stable")
    if eigvals_only:
        return eigenvalues[order]
    return eigenvalues[order], eigenvectors[:, order]
