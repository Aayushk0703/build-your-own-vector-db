# build-your-own-vector-db
# A vector database, from scratch

No Pinecone, no FAISS, no Chroma, no `sklearn.neighbors`. Everything here
is numpy plus plain Python data structures.

## What's in here

| File | What it is |
|---|---|
| `brute_force.py` | Exact cosine-similarity search. The ground truth. |
| `hnsw.py` | HNSW (Hierarchical Navigable Small World) approximate index, built from the Malkov & Yashunin algorithm. |
| `api.py` | A five-method API (`insert`, `bulk_insert`, `search`, `delete`, `len`) over either backend. |
| `data_gen.py` | Synthetic data generator: 30 uneven Gaussian blobs on the unit hypersphere, so the geometry is genuinely clustered/lumpy rather than uniform random noise. |
| `build_chunked.py` | Builds the 50k-vector HNSW index, checkpointing to disk so the build can span multiple runs. |
| `evaluate.py` | Loads the checkpointed index, computes exact ground truth with the brute-force index, and reports recall@10 and query latency across a range of `efSearch` values. |
| `tradeoff_experiment.py` | A smaller, faster experiment isolating how `M` / `efConstruction` trade build cost for recall. |
| `demo.py` | A short, readable walkthrough of insert / search / delete on both backends side by side. |

## How brute force works

Normalize every vector once. Cosine similarity between unit vectors is
just their dot product, so a query becomes one matrix-vector multiply
against the whole stored matrix — numpy hands that to BLAS and it's fast
even at 50,000 vectors (well under 1ms per query here). Deletion is a
tombstone set, compacted (rows physically dropped) once tombstones pass
30% of the index, so a long run of deletes doesn't degrade every query
forever.

## How the HNSW index works

Build several layers of a graph, like a skip list: the top layer has a
handful of nodes and long edges, each layer below is denser, and layer 0
contains every point. A search starts at the top and greedily walks
downhill — always moving to whichever neighbor is closer to the query —
until it can't improve, then drops a layer and repeats. At layer 0 it
switches to a wider beam search (`efSearch` controls the beam width) to
collect the actual top-k. Insertion does almost the same walk, then wires
the new node into every layer it belongs to and prunes any node that now
has too many neighbors.

Two implementation details actually mattered, and both were bugs I hit
and fixed while building this — worth knowing if you're implementing
HNSW yourself, since they're easy to get subtly wrong:

**1. Neighbor selection needs the diversity heuristic, not plain top-M.**
The obvious first implementation links a new node to its M closest
existing neighbors. On uniform random data that's fine. On clustered
data it's a trap: a node's M closest neighbors are almost always in the
*same* cluster, so every node ends up richly connected within its own
cluster and only accidentally bridges to others. The graph looks healthy
by every structural metric (every node at the max degree, mostly
connected) but has too few cross-cluster edges, so greedy descent
sometimes walks into the wrong cluster and can't escape inside the
search budget. In testing, this showed up as recall collapsing to 0%
for some queries while staying high for others — a very different
symptom than uniformly-mediocre recall, and the giveaway that it was a
structural problem, not just under-tuned parameters. The fix is Malkov &
Yashunin's actual selection rule: walk candidates closest-first and only
keep one if it's closer to the new node than to every neighbor already
accepted. That single diversity check is what forces some edges to point
toward genuinely different regions of the space instead of piling up
redundant same-direction connections.

**2. That same heuristic has to be used when *pruning* an overfull
neighbor list, not just when initially selecting neighbors.** I missed
this at first: new nodes got good, diverse neighbor sets, but whenever an
*existing* node accumulated too many reverse-links and had to be pruned
back down to its cap, I pruned by plain nearest-distance. That silently
re-collapses the graph over time — it strips out exactly the long bridge
edges the heuristic had worked to create, re-favoring tight in-cluster
connections as more nodes get inserted and re-pruned. The effect was
dramatic: on a 3,000-vector test, querying with a vector *already in the
index* failed to return itself as the top match (cosine similarity 1.0,
and it still lost). A BFS from the entry point at layer 0 reached only
776 of 3,000 nodes — the graph was fragmented into a large disconnected
island. Using the diversity heuristic for pruning too fixed it: full
connectivity (3,000/3,000 reachable), and self-queries return themselves
exactly.

Deletion is a tombstone (`self.deleted`), not a structural removal. The
node stays in the graph as a bridge so removing it doesn't fragment the
neighborhood around it, but it's filtered out of search results. A real
hard delete means finding every neighbor that pointed at the deleted
node, re-running a local search to find each of them a replacement edge,
and hoping the deleted node wasn't a load-bearing hub whose removal
splits the graph. That's genuinely more work and easy to get subtly
wrong, which is exactly why production graph indexes tombstone-and-
periodically-rebuild instead, same as here.

## Results

**Setup:** 50,000 vectors, 64 dimensions, 30 clusters of very uneven size
and tightness (Dirichlet-weighted cluster sizes, spreads from 0.05 to
0.4). 500 held-out queries from the same distribution, exact top-10
computed by brute force as ground truth. HNSW built with `M=16`,
`efConstruction=100`.

Recall@10 and per-query latency as the search-time beam width (`efSearch`)
grows:

| efSearch | recall@10 | avg query time | vs. brute-force (0.68ms) |
|---:|---:|---:|---:|
| 10  | 0.244 | 0.76 ms  | 0.9x (slower) |
| 20  | 0.379 | 1.04 ms  | 0.7x (slower) |
| 50  | 0.596 | 2.22 ms  | 0.3x (slower) |
| 100 | 0.761 | 3.81 ms  | 0.2x (slower) |
| 200 | 0.895 | 6.38 ms  | 0.1x (slower) |
| 400 | 0.964 | 12.31 ms | 0.1x (slower) |

Two honest observations that the exercise is specifically about:

- **Recall depends on `efSearch` exactly the way the theory says it
  should** — a smooth, monotonic tradeoff, which is the entire point of
  having a tunable search-time knob. You get near-exact results (96%) by
  paying for a wider beam.
- **HNSW is slower than brute force here, at every setting.** That's not
  a bug, it's the actual cost of doing this in pure Python at this scale:
  numpy's brute force is one vectorized BLAS call handling all 50,000
  comparisons at once, while the graph traversal does thousands of tiny
  Python-level heap operations and function calls per query, each with
  real interpreter overhead. Real HNSW libraries (hnswlib, FAISS) win
  because their traversal loop is compiled C/C++; the *algorithm's*
  advantage (roughly logarithmic scaling vs. brute force's linear scan)
  only shows up in wall-clock time once you're either at a scale where
  linear scan is actually slow (millions of vectors, or vectors too large
  to fit densely in cache) or you've paid down the constant-factor
  overhead by writing the inner loop in something faster than Python.
  This implementation makes the algorithmic tradeoff real and
  measurable; it doesn't (and can't, in pure Python) make it a wall-clock
  win at 50k vectors.

A separate smaller experiment (`tradeoff_experiment.py`, 8,000 vectors)
isolates the build-time side of the same tradeoff — recall and build
cost both climb together as `M`/`efConstruction` increase, since a
denser, better-explored graph takes longer to construct but navigates
better once built.

One more scale-dependent fact worth having: recall for a *fixed*
`M`/`efConstruction` degrades as the dataset grows. At `M=16,
efConstruction=100`, an 8,000-vector index gets ~98% recall@10 at
`efSearch=100`; the same parameters on 50,000 vectors get 76%. Making
recall roughly constant as data grows means growing `M`/`efConstruction`
with it — the graph parameters need to be sized to the dataset, not
just to the k you care about.

## Running it
MAC

python3 -m vectordb.demo                  # quick illustrated walkthrough
python3 -m vectordb.tradeoff_experiment   # M/efConstruction tradeoff, ~2 min
python3 -m vectordb.build_chunked         # builds the 50k HNSW index (run repeatedly to resume)
python3 -m vectordb.evaluate              # recall@10 + speed table against ground truth

Running it

pip install -r requirements.txt

Windows

python -m vectordb.demo

python -m vectordb.tradeoff_experiment

python -m vectordb.build_chunked

python -m vectordb.evaluate


