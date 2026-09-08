"""
Synthetic but non-trivial vector data: a mixture of Gaussian blobs on the
unit hypersphere, with different cluster sizes and different within-cluster
spreads. Random vectors are nearly all pairwise-orthogonal in high dimension
(boring, uniform geometry, and unrealistically easy for an approximate
index). Real embeddings instead sit in dense, uneven clumps -- HNSW's whole
value proposition is exploiting that lumpiness, so the test data should
have it.
"""
import numpy as np


def generate_clustered_vectors(n_vectors, dim, n_clusters=30, seed=0):
    rng = np.random.default_rng(seed)

    # cluster centers, spread out on the sphere
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # uneven cluster sizes (some big popular topics, some rare ones).
    # Largest-remainder method: guarantees non-negative counts that sum
    # exactly to n_vectors (naive "+= drift on the last bucket" can push
    # a small cluster negative).
    weights = rng.dirichlet(np.full(n_clusters, 0.5))
    raw = weights * n_vectors
    counts = np.floor(raw).astype(int)
    remainder = n_vectors - counts.sum()
    fractional_order = np.argsort(-(raw - counts))  # biggest leftover fraction first
    counts[fractional_order[:remainder]] += 1

    # uneven cluster tightness
    spreads = rng.uniform(0.05, 0.4, size=n_clusters)

    vectors = np.empty((n_vectors, dim), dtype=np.float32)
    cluster_ids = np.empty(n_vectors, dtype=int)
    row = 0
    for c in range(n_clusters):
        n = counts[c]
        if n == 0:
            continue
        noise = rng.normal(scale=spreads[c], size=(n, dim)).astype(np.float32)
        v = centers[c] + noise
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        vectors[row:row + n] = v
        cluster_ids[row:row + n] = c
        row += n

    return vectors, cluster_ids
