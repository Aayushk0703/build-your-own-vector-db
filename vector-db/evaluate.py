import pickle
import time
import numpy as np

from vectordb.brute_force import BruteForceIndex
from vectordb.data_gen import generate_clustered_vectors

DIM = 64
N = 50_000
N_QUERIES = 500
K = 10
N_CLUSTERS = 30
SEED = 42
CKPT = "/home/claude/hnsw_ckpt.pkl"


def main():
    print(f"Generating {N} vectors (dim={DIM}, {N_CLUSTERS} clusters)...")
    vecs, _ = generate_clustered_vectors(N, DIM, n_clusters=N_CLUSTERS, seed=SEED)
    ids = list(range(N))
    queries, _ = generate_clustered_vectors(N_QUERIES, DIM, n_clusters=N_CLUSTERS, seed=SEED + 1)

    print("Building brute-force (ground truth) index...")
    t0 = time.perf_counter()
    bf = BruteForceIndex(DIM)
    bf.insert(ids, vecs)
    print(f"  build time: {time.perf_counter() - t0:.2f}s")

    print(f"Computing exact top-{K} for {N_QUERIES} queries (ground truth)...")
    t0 = time.perf_counter()
    ground_truth = [set(i for i, _ in bf.search(q, K)) for q in queries]
    bf_query_time = (time.perf_counter() - t0) / N_QUERIES
    print(f"  avg brute-force query time: {bf_query_time * 1000:.3f} ms")

    with open(CKPT, "rb") as f:
        hnsw, done = pickle.load(f)
    assert done >= N, f"HNSW build incomplete: {done}/{N}"
    print(f"Loaded HNSW index: {len(hnsw)} vectors, M={hnsw.M}, "
          f"efConstruction={hnsw.ef_construction}, max_level={hnsw.max_level}")

    print("\nRecall@10 and speed vs efSearch:")
    print(f"{'ef':>6} | {'recall@10':>10} | {'avg query ms':>13} | {'speedup vs brute-force':>22}")
    print("-" * 62)
    for ef in [10, 20, 50, 100, 200, 400]:
        t0 = time.perf_counter()
        recalls = []
        for q, gt in zip(queries, ground_truth):
            got = set(i for i, _ in hnsw.search(q, K, ef=ef))
            recalls.append(len(got & gt) / K)
        elapsed = time.perf_counter() - t0
        avg_query_ms = (elapsed / N_QUERIES) * 1000
        speedup = bf_query_time / (elapsed / N_QUERIES)
        print(f"{ef:>6} | {np.mean(recalls):>10.3f} | {avg_query_ms:>13.4f} | {speedup:>21.1f}x")

    print("\nDeletion check: deleting 1000 random ids from HNSW, confirming they vanish from results...")
    rng = np.random.default_rng(0)
    to_delete = set(int(x) for x in rng.choice(ids, size=1000, replace=False))
    for d in to_delete:
        hnsw.delete(d)
    leaked = 0
    for q in queries[:100]:
        got = [i for i, _ in hnsw.search(q, K, ef=200)]
        leaked += sum(1 for g in got if g in to_delete)
    print(f"  deleted ids appearing in 100 post-delete queries' top-10: {leaked} (should be 0)")


if __name__ == "__main__":
    main()
