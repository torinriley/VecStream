# HNSW design

HNSW approximates nearest-neighbor search with a hierarchy of proximity graphs. Sparse upper layers provide long jumps; dense level 0 contains every point and provides local refinement.

```text
level 2       A ---------------- D
              |                  |
level 1       A ------ B ------- D
              |        |         |
level 0       A -- B -- C -- E -- D -- F
```

For `M > 1`, VecStream draws a uniform `u` and assigns `level = floor(-ln(u) / ln(M))`. Thus `P(level >= l) = M^-l`: every successively higher layer is exponentially smaller. A supplied seed makes levels and graph topology reproducible.

Insertion starts at the highest-layer entry point. Greedy searches with `ef=1` descend to the new node's maximum level. At each participating layer, a best-first search keeps `ef_construction` candidates, then selects at most `M` neighbors (`2M` at level 0). Connections are reciprocal.

Nearest-M tends to choose a tight cluster of nearly collinear points. VecStream applies the HNSW diversity heuristic: in distance order, accept a candidate only if it is closer to the query than to every already-selected neighbor. Rejected points fill unused capacity. The same rule prunes overflowing adjacency lists. This costs pairwise comparisons within the bounded candidate set but retains links in different directions.

Querying greedily descends upper layers, then performs a best-first level-0 search bounded by `ef_search`. Increasing it normally improves recall at the cost of latency.

Graph vectors are normalized at insertion, making cosine distance `1 - dot(a, b)`. Zero vectors normalize to zero and therefore have similarity zero.

`validate()` checks entry-point, node/level membership, edge target, reciprocal-edge, degree, and finite-value invariants. Level-0 degree is at most `2M`; upper-layer degree is at most `M`.

The graph requires vector memory `O(Nd)` and adjacency memory approximately `O(NM)`. Expected search behavior is logarithmic on well-formed data, but HNSW has no worst-case logarithmic guarantee and quality depends on parameters, distribution, and insertion order.
