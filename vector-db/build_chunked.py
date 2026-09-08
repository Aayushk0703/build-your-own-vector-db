import pickle
import time
import os
import sys

from vectordb.hnsw import HNSWIndex
from vectordb.data_gen import generate_clustered_vectors

DIM = 64
N = 50_000
N_CLUSTERS = 30
SEED = 42
CKPT = "/home/claude/hnsw_ckpt.pkl"
CHUNK_SECONDS = 200


def main():
    vecs, _ = generate_clustered_vectors(N, DIM, n_clusters=N_CLUSTERS, seed=SEED)
    ids = list(range(N))

    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            hnsw, next_i = pickle.load(f)
        print(f"resuming from i={next_i}")
    else:
        hnsw = HNSWIndex(DIM, M=16, ef_construction=100, ef_search=50, seed=SEED)
        next_i = 0

    t0 = time.perf_counter()
    i = next_i
    while i < N:
        hnsw.insert(ids[i], vecs[i])
        i += 1
        if time.perf_counter() - t0 > CHUNK_SECONDS:
            break

    with open(CKPT, "wb") as f:
        pickle.dump((hnsw, i), f)

    print(f"progress: {i}/{N}  ({time.perf_counter()-t0:.1f}s this chunk)")
    if i >= N:
        print("DONE")


if __name__ == "__main__":
    main()
