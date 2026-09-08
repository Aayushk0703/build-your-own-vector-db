"""
HNSW (Hierarchical Navigable Small World) index, built from scratch.

Idea in one paragraph: build several layers of a proximity graph, stacked
like a skip list. The top layer has very few nodes and long-range edges,
so a search there jumps across the whole space in a handful of hops.
Each lower layer is denser. A search starts at the top, greedily walks
downhill (always moving to whichever neighbor is closer to the query)
until it can't improve, then drops one layer and repeats, ending with a
wider beam search at layer 0 to collect the actual top-k. Insertion does
almost the same walk, then wires the new node into each layer it belongs
to, and prunes anyone who now has too many neighbors.

This follows Malkov & Yashunin (2016/2018):
  - Neighbor selection uses their diversity heuristic (SELECT-NEIGHBORS-
    HEURISTIC), not plain top-M-by-distance. This matters more than it
    sounds: top-M naturally links a new node to its M closest existing
    neighbors, which on clustered data means "the M closest points in the
    same cluster" almost every time. Every node ends up richly connected
    *within* its own cluster and only accidentally connected to
    neighboring clusters. The result is a graph that looks fine (uniform
    degree, mostly connected) but has too few bridges between clusters,
    so greedy descent can walk into the wrong cluster and then can't get
    out inside the search budget -- recall for those queries craters to
    near zero while unaffected queries stay high, which is exactly what
    plain top-M selection produced in testing here. The heuristic instead
    only keeps a candidate neighbor if it is closer to the new node than
    it is to any neighbor already accepted -- which prunes redundant
    same-direction edges and preferentially keeps the bridges. Concretely
    this fixed the clustered-data recall failure in this implementation
    (see README for before/after numbers).
  - Distances are cosine distance (1 - dot product) on unit vectors.
"""
import heapq
import math
import random

import numpy as np


class HNSWIndex:
    def __init__(self, dim, M=16, ef_construction=100, ef_search=50, seed=0):
        self.dim = dim
        self.M = M                      # max neighbors per node, layers > 0
        self.M0 = 2 * M                 # max neighbors at layer 0 (denser base layer)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)     # level-generation multiplier

        self.vectors = {}               # id -> normalized np.array
        self.neighbors = {}             # id -> {layer: set(neighbor ids)}
        self.levels = {}                # id -> top layer of this node
        self.deleted = set()            # tombstones
        self.entry_point = None
        self.max_level = -1

        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.vectors) - len(self.deleted)

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _dist(self, a, b):
        return 1.0 - float(np.dot(a, b))

    def _random_level(self):
        return int(-math.log(self._rng.random()) * self.mL)

    # ---------- core graph search primitives ----------

    def _greedy_descend(self, query, entry_id, entry_dist, layer):
        """Walk downhill within a single layer until no neighbor improves distance."""
        cur_id, cur_dist = entry_id, entry_dist
        improved = True
        while improved:
            improved = False
            for nb in self.neighbors[cur_id].get(layer, ()):
                d = self._dist(query, self.vectors[nb])
                if d < cur_dist:
                    cur_dist, cur_id, improved = d, nb, True
        return cur_id, cur_dist

    def _search_layer(self, query, entry_points, ef, layer):
        """Beam search within one layer. Returns list of (id, dist), best first."""
        visited = set(entry_points)
        candidates = []  # min-heap of (dist, id): frontier to expand
        results = []     # max-heap of (-dist, id): best `ef` found so far

        for ep in entry_points:
            d = self._dist(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            cur_dist, cur_id = heapq.heappop(candidates)
            worst_kept = -results[0][0]
            if cur_dist > worst_kept and len(results) >= ef:
                break  # nothing left in the frontier can beat our current worst kept result
            for nb in self.neighbors[cur_id].get(layer, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist(query, self.vectors[nb])
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted([(nid, -negd) for negd, nid in results], key=lambda x: x[1])

    def _select_neighbors(self, candidates, m):
        """
        Diversity-aware neighbor selection (paper's SELECT-NEIGHBORS-HEURISTIC,
        simplified: no extendCandidates/keepPrunedConnections passes).

        `candidates` is sorted ascending by distance to the new node. Walk it
        in that order and keep a candidate only if it's closer to the new
        node than it is to every neighbor already accepted. That single
        check is what prevents the selected set from being M near-duplicates
        all pointing into the same cluster -- it forces at least some edges
        to point toward genuinely different regions of the space.
        """
        selected = []
        for cand_id, cand_dist in candidates:
            if len(selected) >= m:
                break
            is_diverse = True
            for sel_id, _ in selected:
                if self._dist(self.vectors[cand_id], self.vectors[sel_id]) < cand_dist:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append((cand_id, cand_dist))

        # if the diversity filter was too strict and left us under m neighbors,
        # top up with the next-closest rejected candidates rather than leave
        # a node under-connected.
        if len(selected) < m:
            selected_ids = set(nid for nid, _ in selected)
            for cand_id, cand_dist in candidates:
                if len(selected) >= m:
                    break
                if cand_id not in selected_ids:
                    selected.append((cand_id, cand_dist))
                    selected_ids.add(cand_id)
        return selected

    # ---------- public API ----------

    def insert(self, id_, vector):
        v = self._normalize(vector)
        level = self._random_level()

        if self.entry_point is None:
            self.vectors[id_] = v
            self.levels[id_] = level
            self.neighbors[id_] = {l: set() for l in range(level + 1)}
            self.entry_point = id_
            self.max_level = level
            return

        cur, cur_dist = self.entry_point, self._dist(v, self.vectors[self.entry_point])
        for lc in range(self.max_level, level, -1):
            cur, cur_dist = self._greedy_descend(v, cur, cur_dist, lc)

        self.vectors[id_] = v
        self.neighbors[id_] = {}
        ep = [cur]
        for lc in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(v, ep, self.ef_construction, lc)
            m = self.M0 if lc == 0 else self.M
            selected = self._select_neighbors(candidates, m)
            self.neighbors[id_][lc] = set(nid for nid, _ in selected)
            ep = [nid for nid, _ in candidates]

            # wire the edge back, then prune the neighbor if it's now overfull.
            # Crucially, use the same diversity heuristic here, not plain
            # nearest-M -- pruning by raw distance alone was the actual bug:
            # it strips away exactly the long bridging edges the heuristic
            # worked to create, re-collapsing the graph toward tight,
            # same-cluster-only connections as more nodes accumulate and
            # get re-pruned over time. That fragmented the graph badly
            # (only ~25% of nodes reachable from the entry point in testing)
            # even though every node still had close to the max degree.
            for nid, _ in selected:
                self.neighbors[nid].setdefault(lc, set()).add(id_)
                if len(self.neighbors[nid][lc]) > m:
                    cand_pairs = sorted(
                        ((cid, self._dist(self.vectors[nid], self.vectors[cid]))
                         for cid in self.neighbors[nid][lc]),
                        key=lambda x: x[1],
                    )
                    kept = self._select_neighbors(cand_pairs, m)
                    self.neighbors[nid][lc] = set(cid for cid, _ in kept)

        self.levels[id_] = level
        if level > self.max_level:
            self.max_level = level
            self.entry_point = id_

    def search(self, query, k=10, ef=None):
        if self.entry_point is None:
            return []
        ef = ef if ef is not None else max(self.ef_search, k)
        q = self._normalize(query)

        cur, cur_dist = self.entry_point, self._dist(q, self.vectors[self.entry_point])
        for lc in range(self.max_level, 0, -1):
            cur, cur_dist = self._greedy_descend(q, cur, cur_dist, lc)

        candidates = self._search_layer(q, [cur], ef, 0)
        candidates = [(nid, d) for nid, d in candidates if nid not in self.deleted]
        candidates.sort(key=lambda x: x[1])
        return [(nid, 1.0 - d) for nid, d in candidates[:k]]  # back to cosine similarity

    def delete(self, id_):
        """
        Soft delete only. The node stays in the graph as a 'bridge' so the
        graph doesn't lose connectivity around it, but it's filtered out of
        search results. This is the honest answer to "delete in a graph
        index": a *hard* delete means finding every neighbor that pointed
        at this node, re-linking each of them to a new candidate (often by
        re-running a local search from their remaining neighbors), and
        hoping you don't fragment the graph into an unreachable island if
        the deleted node was a load-bearing hub. That's real work and it's
        easy to get subtly wrong, so most production graph indexes
        (including this one) tombstone instead and periodically rebuild.
        """
        if id_ in self.vectors:
            self.deleted.add(id_)
