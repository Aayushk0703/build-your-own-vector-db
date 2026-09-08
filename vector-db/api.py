"""
A small, boring API over both index types so you can swap one for the
other without touching calling code.
"""
from .brute_force import BruteForceIndex
from .hnsw import HNSWIndex


class VectorDB:
    def __init__(self, dim, backend="hnsw", **backend_kwargs):
        if backend == "brute_force":
            self._index = BruteForceIndex(dim)
        elif backend == "hnsw":
            self._index = HNSWIndex(dim, **backend_kwargs)
        else:
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend

    def insert(self, id_, vector):
        if self.backend == "brute_force":
            self._index.insert([id_], vector[None, :])
        else:
            self._index.insert(id_, vector)

    def bulk_insert(self, ids, vectors):
        if self.backend == "brute_force":
            self._index.insert(ids, vectors)
        else:
            for id_, v in zip(ids, vectors):
                self._index.insert(id_, v)

    def search(self, query, k=10, **kwargs):
        return self._index.search(query, k=k, **kwargs)

    def delete(self, id_):
        self._index.delete(id_)

    def __len__(self):
        return len(self._index)
