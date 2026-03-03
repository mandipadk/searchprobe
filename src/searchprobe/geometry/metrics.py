"""Geometric metrics for embedding space analysis."""

import math

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(vec_a: NDArray[np.float32], vec_b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def angular_distance(vec_a: NDArray[np.float32], vec_b: NDArray[np.float32]) -> float:
    """Compute angular distance between two vectors.

    Angular distance = arccos(cosine_similarity) / pi, normalized to [0, 1].

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        Angular distance in [0, 1]
    """
    sim = cosine_similarity(vec_a, vec_b)
    # Clamp to valid range for arccos
    sim = max(-1.0, min(1.0, sim))
    return float(math.acos(sim) / math.pi)


def local_intrinsic_dimensionality(
    embeddings: NDArray[np.float32],
    k: int = 10,
) -> float:
    """Estimate local intrinsic dimensionality using Maximum Likelihood Estimation.

    Based on Amsaleg et al. 2015 - "Estimating Local Intrinsic Dimensionality".
    Lower LID means queries are clustered in a lower-dimensional subspace,
    indicating the embedding model doesn't differentiate well in this region.

    Args:
        embeddings: Matrix of shape (n_samples, embedding_dim)
        k: Number of nearest neighbors to use

    Returns:
        MLE estimate of local intrinsic dimensionality
    """
    from sklearn.neighbors import NearestNeighbors

    n_samples = embeddings.shape[0]
    if n_samples < k + 1:
        k = max(1, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # Remove self-distance (first column)
    distances = distances[:, 1:]

    # Avoid log(0)
    distances = np.maximum(distances, 1e-10)

    # MLE estimate: LID = -1 / (1/k * sum(log(d_i / d_k)))
    lids = []
    for i in range(n_samples):
        d_k = distances[i, -1]
        if d_k <= 0:
            continue
        log_ratios = np.log(distances[i] / d_k)
        mean_log_ratio = np.mean(log_ratios)
        if mean_log_ratio < 0:
            lid = -1.0 / mean_log_ratio
            lids.append(lid)

    return float(np.mean(lids)) if lids else 0.0


def isotropy_score(embeddings: NDArray[np.float32]) -> float:
    """Measure how uniformly distributed embeddings are in the space.

    Based on Mu & Viswanath 2018 - "All-but-the-Top: Simple and Effective
    Postprocessing for Word Representations".

    A perfectly isotropic distribution has score ~1.0.
    Highly anisotropic (clustered in few directions) has score near 0.

    Args:
        embeddings: Matrix of shape (n_samples, embedding_dim)

    Returns:
        Isotropy score in [0, 1]
    """
    if embeddings.shape[0] < 2:
        return 0.0

    # Center embeddings
    centered = embeddings - np.mean(embeddings, axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability

    if eigenvalues.sum() == 0:
        return 0.0

    # Normalize to get fraction of variance
    eigenvalues_norm = eigenvalues / eigenvalues.sum()

    # Compute entropy-based isotropy
    # Max entropy = log(d) for d dimensions
    nonzero = eigenvalues_norm[eigenvalues_norm > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(len(eigenvalues_norm))

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


def anisotropy_components(
    embeddings: NDArray[np.float32], n_components: int = 5
) -> list[tuple[int, float]]:
    """Identify the dominant directions of anisotropy.

    Args:
        embeddings: Matrix of shape (n_samples, embedding_dim)
        n_components: Number of top components to return

    Returns:
        List of (component_index, variance_fraction) tuples
    """
    if embeddings.shape[0] < 2:
        return []

    centered = embeddings - np.mean(embeddings, axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    if total_var == 0:
        return []

    # Sort descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    components = []
    for i in range(min(n_components, len(sorted_idx))):
        idx = int(sorted_idx[i])
        frac = float(eigenvalues[idx] / total_var)
        components.append((idx, frac))

    return components


def pairwise_cosine_similarities(embeddings: NDArray[np.float32]) -> NDArray[np.float64]:
    """Compute all pairwise cosine similarities.

    Args:
        embeddings: Matrix of shape (n_samples, embedding_dim)

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)
