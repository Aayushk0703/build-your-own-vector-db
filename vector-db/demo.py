"""
A small, readable demo of the API: build a tiny corpus, run a query
against both indexes side by side, insert a fresh vector, delete
something, and show that all three operations behave as expected.

Run: python3 -m vectordb.demo
"""
import numpy as np

from vectordb.api import VectorDB
from vectordb.data_gen import generate_clustered_vectors


def show(label, results):
    print(f"  {label}:")
    for id_, score in results:
        print(f"    id={id_:5d}  cosine_sim={score:.4f}")


def main():
    DIM = 32
    N = 3000
    vecs, cluster_ids = generate_clustered_vectors(N, DIM, n_clusters=8, seed=3)
    ids = list(range(N))

    print(f"Building a brute-force index and an HNSW index over {N} vectors...")
    bf = VectorDB(DIM, backend="brute_force")
    bf.bulk_insert(ids, vecs)

    hnsw = VectorDB(DIM, backend="hnsw", M=16, ef_construction=100, ef_search=100, seed=3)
    hnsw.bulk_insert(ids, vecs)

    query = vecs[42]
    print(f"\nQuery: vector #42 (cluster {cluster_ids[42]})")
    show("brute-force top-5 (exact)", bf.search(query, k=5))
    show("HNSW top-5 (approximate)", hnsw.search(query, k=5, ef=100))

    print("\nInserting a brand-new vector (id=99999)...")
    new_vec = vecs[42] + np.random.default_rng(0).normal(scale=0.02, size=DIM).astype(np.float32)
    bf.insert(99999, new_vec)
    hnsw.insert(99999, new_vec)
    show("brute-force top-5 after insert", bf.search(query, k=5))
    show("HNSW top-5 after insert", hnsw.search(query, k=5, ef=100))
    print("  (the new near-duplicate vector shows up right near the top on both)")

    top_id = bf.search(query, k=1)[0][0]
    print(f"\nDeleting id={top_id} (the current best match) from both indexes...")
    bf.delete(top_id)
    hnsw.delete(top_id)
    show("brute-force top-5 after delete", bf.search(query, k=5))
    show("HNSW top-5 after delete", hnsw.search(query, k=5, ef=100))
    print(f"  id={top_id} is gone from both result sets, as expected.")

    print(f"\nFinal sizes -- brute-force: {len(bf)}, hnsw: {len(hnsw)}")


if __name__ == "__main__":
    main()
