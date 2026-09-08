"""
Isolate the thing this exercise is actually about: what does the
approximation cost you? Build several HNSW indexes with different
(M, efConstruction) settings on the same data, and show recall,
build time, and memory (edges stored) for each. Uses a smaller N so
each configuration builds in a reasonable time.
"""
import time
import numpy as np

from vectordb.brute_force import BruteForceIndex
from vectordb.hnsw import HNSWIndex
from vectordb.data_gen import generate_clustered_vectors

DIM = 64
N = 8000
N_QUERIES = 200
K = 10
N_CLUSTERS = 30
SEED = 7


def main():
    vecs, _ = generate_clustered_vectors(N, DIM, n_clusters=N_CLUSTERS, seed=SEED)
    ids = list(range(N))
    queries, _ = generate_clustered_vectors(N_QUERIES, DIM, n_clusters=N_CLUSTERS, seed=SEED + 1)

    bf = BruteForceIndex(DIM)
    bf.insert(ids, vecs)
    ground_truth = [set(i for i, _ in bf.search(q, K)) for q in queries]

    configs = [(8, 50), (16, 100), (24, 150), (32, 200)]
    print(f"{'M':>4} | {'efConstruction':>14} | {'build s':>8} | {'ms/insert':>10} | "
          f"{'avg edges/node':>15} | {'recall@10 (ef=100)':>20}")
    print("-" * 90)
    for M, efc in configs:
        hnsw = HNSWIndex(DIM, M=M, ef_construction=efc, seed=SEED)
        t0 = time.perf_counter()
        for i, v in zip(ids, vecs):
            hnsw.insert(i, v)
        build_time = time.perf_counter() - t0

        total_edges = sum(len(layers.get(0, ())) for layers in hnsw.neighbors.values())
        avg_edges = total_edges / N

        recalls = []
        for q, gt in zip(queries, ground_truth):
            got = set(i for i, _ in hnsw.search(q, K, ef=100))
            recalls.append(len(got & gt) / K)

        print(f"{M:>4} | {efc:>14} | {build_time:>8.2f} | {build_time/N*1000:>10.3f} | "
              f"{avg_edges:>15.1f} | {np.mean(recalls):>20.3f}")


if __name__ == "__main__":
    main()
