"""
Brute-force exact nearest-neighbor index.

This is the ground truth. No cleverness: normalize every vector, keep them
in one dense matrix, and on every query multiply the whole matrix by the
query vector. Cosine similarity between unit vectors is just their dot
product, so this reduces to a single matrix-vector multiply per query,
which numpy does in optimized C/BLAS code.

Deletion is handled with a "tombstone" set rather than shifting rows out
of the matrix on every delete (that would be O(n) per delete). Tombstones
are compacted (rows physically removed) once they build up past 30% of
the index, so search cost doesn't creep up forever.
"""
try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is not installed. Please install it using 'pip install numpy'.")


class BruteForceIndex:
    def __init__(self, dim):
        self.dim = dim
        self._ids = []            # row i -> external id
        self._id_to_row = {}      # external id -> row i
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._deleted = set()

    def __len__(self):
        return len(self._ids) - len(self._deleted)

    @staticmethod
    def _normalize(vecs):
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms

    def insert(self, ids, vectors):
        """ids: list of hashable ids. vectors: (n, dim) array-like."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
            ids = [ids]
        vectors = self._normalize(vectors)
        for _id in ids:
            self._id_to_row[_id] = len(self._ids)
            self._ids.append(_id)
        self._vectors = (np.vstack([self._vectors, vectors])
                          if self._vectors.size else vectors)

    def delete(self, id_):
        if id_ in self._id_to_row:
            self._deleted.add(id_)

    def _compact_if_needed(self):
        if len(self._deleted) > 0.3 * max(1, len(self._ids)):
            keep = [i for i, _id in enumerate(self._ids) if _id not in self._deleted]
            self._vectors = self._vectors[keep]
            self._ids = [self._ids[i] for i in keep]
            self._id_to_row = {v: i for i, v in enumerate(self._ids)}
            self._deleted.clear()

    def search(self, query, k=10):
        self._compact_if_needed()
        if len(self._ids) == 0:
            return []
        q = self._normalize(np.asarray(query, dtype=np.float32)[None, :])[0]
        sims = self._vectors @ q  # cosine similarity to every stored vector, in one shot
        if self._deleted:
            for _id in self._deleted:
                row = self._id_to_row.get(_id)
                if row is not None:
                    sims[row] = -np.inf
        k = min(k, len(self))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._ids[i], float(sims[i])) for i in top]
