# Methods

## 1. Setting
A directed road graph `G = (V, E)` with free-flow times `tau0_e`. Cameras sit at a subset `S` of nodes
and detect every vehicle that passes the node, recording the time, the incoming edge, the outgoing edge
and an appearance embedding. A passage at node `n` is the instant a vehicle *enters* the edge leaving
`n`; for a path `P = (e_1, ..., e_k)` from camera `u` to camera `v` the observed time `T = t_v - t_u`
equals `sum_{e in P} tau_e` (edge time = entry to the next edge minus entry to the edge, junction delay
included).

## 2. Unique paths
In a grid, two nodes on a common row or column have exactly one shortest path. Only such *straight*
camera pairs are used, which removes route ambiguity between cameras (vehicles follow shortest routes,
and sub-paths of shortest paths are shortest). A candidate at `u` must leave along the first edge of
the path, a candidate at `v` must arrive along the last edge.

## 3. Association
For a pair `(u, v)` with free-flow path time `T0`: candidates `(i, j)` are feasible if
`0.7 T0 <= t_j - t_i <= 6 T0 + 20 s` and the cosine similarity `c_ij` of the embeddings is at least 0.45.
The cost is `(1 - c_ij) + 0.1 * clip((t_j - t_i)/T0 - 1, 0) / 6`. The assignment is solved with the
Hungarian algorithm on a matrix augmented with dummy rows and columns of cost `(1 - 0.45) + 0.1`, so that
vehicles that leave the corridor stay unmatched. Baselines: *appearance only* (no window beyond causality)
and *FIFO* (window and first-in-first-out order, no appearance).

## 4. Slice statistics
For each slice (600 s) and pair with at least 3 matches, the travel time is the mean of the matched
`t_j - t_i`, clipped above its 95th percentile (the mean is the statistic that adds up over edges;
clipping resists wrong matches).

## 5. Network tomography
Unknowns are congestion factors `rho_e >= 1` with `tau_e = tau0_e rho_e`. Each observation gives
`sum_{e in P} (tau0_e / T0) rho_e = T / T0` with weight equal to its number of matches `w_k`.
The estimate solves

`(A^T W A + lambda L + mu I) rho = A^T W y + mu rho_prior`,

where `L` is the Laplacian of the edge adjacency graph (edges sharing a node are adjacent), `lambda = 0.5`,
`mu = 0.02`, `rho_prior` is the median observed ratio, and the solution is clipped below at 1.

## 6. Placement
`covered(S)` is the union of edges on straight paths between pairs in `S`. Greedy placement adds the node
with the largest marginal gain in `|covered(S)|`. Baselines: the `k` nodes closest to the centroid and
random sets (5 draws, averaged).

## 7. Spanner sensitivity
The greedy `t`-spanner (`t = 3.0`) is built on a weight vector (oracle, tomography estimate, free flow)
by scanning edges in ascending weight and adding an edge when the current spanner distance exceeds
`t * w_e`. The result is evaluated on the *true* weights: the fraction of ordered node pairs whose spanner
distance exceeds `t` times the graph distance (violation fraction) and the maximum dilation.

## 8. Metrics
Precision = correct matches / matches; recall = correct matches / vehicles detected at both ends in the
right order. Edge MAPE = mean `|tau_hat - tau| / tau` over (edge, slice) pairs with at least 3 true
traversals, reported for edges covered and not covered by the cameras. Seeds: traffic seeds 0-2; the
tomography regularisation and the slice statistic were set on separate development seeds (100, 101).
