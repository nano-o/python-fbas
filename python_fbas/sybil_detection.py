from __future__ import annotations

from collections import defaultdict
import logging
import math
from typing import Final, Iterable

import networkx as nx

DEFAULT_STEPS: Final[int] = 5
DEFAULT_CAPACITY: Final[float] = 1.0
DEFAULT_TRUSTRANK_ALPHA: Final[float] = 0.2
DEFAULT_TRUSTRANK_EPSILON: Final[float] = 1e-8
DEFAULT_MAXFLOW_SEED_CAPACITY: Final[float] = 1.0
DEFAULT_MAXFLOW_MODE: Final[str] = "standard"
MAXFLOW_MODES: Final[tuple[str, ...]] = ("standard",)
DEFAULT_BIMODALITY_THRESHOLD: Final[float] = 5 / 9
DEFAULT_MAXFLOW_SWEEP_POST_THRESHOLD_STEPS: Final[int] = 0
DEFAULT_TOP_TIER_LAMBDA: Final[float] = 1.0
DEFAULT_TOP_TIER_SCORE_THRESHOLD: Final[float] = 0.65


def compute_trust_scores(
    graph: nx.DiGraph,
    seeds: str | Iterable[str],
    *,
    capacity: float = DEFAULT_CAPACITY,
    steps: int = DEFAULT_STEPS,
) -> tuple[dict[str, float], list[float], list[float]]:
    """
    Compute capacity-limited trust scores from a set of trusted orgs.

    Trust mass starts at each seed (1.0 per seed) and is propagated along
    outgoing edges for a fixed number of steps. Each node may send at most
    `capacity` total mass per step, split evenly among its outgoing neighbors.
    Any remaining mass stays at the node. Scores are the cumulative mass
    received over all steps, including the initial seed mass.

    This design intentionally uses a fixed number of steps with cumulative
    scoring. A fix-point is only meaningful for steady-state mass, which would
    require decay/restart and tends to leak more trust across small cuts.
    Bounded walks keep trust local, which is the desired sybil-resistance
    behavior.
    """
    if isinstance(seeds, str):
        seed_list = [seeds]
    else:
        seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    missing = [seed for seed in seed_list if seed not in graph]
    if missing:
        raise ValueError(f"seeds not found in graph: {missing}")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    scores: dict[str, float] = {node: 0.0 for node in graph.nodes}
    mass: dict[str, float] = defaultdict(float)
    for seed in dict.fromkeys(seed_list):
        scores[seed] += 1.0
        mass[seed] += 1.0

    for _ in range(steps):
        next_mass: dict[str, float] = defaultdict(float)
        for node, amount in mass.items():
            if amount <= 0:
                continue
            successors = list(graph.successors(node))
            if not successors:
                next_mass[node] += amount
                continue

            send_amount = min(capacity, amount)
            remaining = amount - send_amount
            share = send_amount / len(successors)
            for succ in successors:
                next_mass[succ] += share
            if remaining > 0:
                next_mass[node] += remaining

        for node, amount in next_mass.items():
            scores[node] += amount
        mass = next_mass

    return scores


def compute_trustrank_scores(
    graph: nx.DiGraph,
    seeds: str | Iterable[str],
    *,
    alpha: float = DEFAULT_TRUSTRANK_ALPHA,
    epsilon: float = DEFAULT_TRUSTRANK_EPSILON,
    max_iters: int = 1000,
) -> dict[str, float]:
    """
    Compute TrustRank scores via a personalized PageRank fixpoint.

    TrustRank is a random walk with restart: with probability `alpha` the walk
    jumps back to the seed distribution, otherwise it follows outgoing edges
    uniformly. The iteration runs until the L1 delta between successive score
    vectors is at most `epsilon` (or `max_iters` is reached).

    TODO We have a normalization problem: sybils in a small sybil cluster might
    have high rank just because it's few nodes.
    """
    if isinstance(seeds, str):
        seed_list = [seeds]
    else:
        seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    missing = [seed for seed in seed_list if seed not in graph]
    if missing:
        raise ValueError(f"seeds not found in graph: {missing}")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")

    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}

    unique_seeds = list(dict.fromkeys(seed_list))
    seed_weight = 1.0 / len(unique_seeds)
    teleport = [0.0] * n
    for seed in unique_seeds:
        teleport[index[seed]] = seed_weight

    scores = teleport[:]
    for _ in range(max_iters):
        next_scores = [alpha * value for value in teleport]
        for node in nodes:
            node_index = index[node]
            out_degree = graph.out_degree(node)
            if out_degree == 0:
                dangling_mass = (1.0 - alpha) * scores[node_index]
                if dangling_mass:
                    for i in range(n):
                        next_scores[i] += dangling_mass * teleport[i]
                continue

            share = (1.0 - alpha) * scores[node_index] / out_degree
            if share == 0:
                continue
            for succ in graph.successors(node):
                next_scores[index[succ]] += share

        delta = sum(abs(next_scores[i] - scores[i]) for i in range(n))
        scores = next_scores
        if delta <= epsilon:
            break

    return {node: scores[index[node]] for node in nodes}


def compute_bimodality_coefficient(values: Iterable[float]) -> float:
    values_list = list(values)
    n = len(values_list)
    if n < 4:
        return 0.0
    mean = sum(values_list) / n
    m2 = sum((value - mean) ** 2 for value in values_list) / n
    if m2 <= 1e-12:
        return 0.0
    m3 = sum((value - mean) ** 3 for value in values_list) / n
    m4 = sum((value - mean) ** 4 for value in values_list) / n
    g1 = m3 / (m2 ** 1.5)
    g2 = m4 / (m2 ** 2) - 3.0
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g1 ** 2 + 1) / (g2 + correction)


def compute_top_tier_score(
    graph: nx.DiGraph,
    *,
    lambda_: float = DEFAULT_TOP_TIER_LAMBDA,
) -> float:
    """
    Compute a top-tier score for a digraph based on missing arc regularity.

    The score treats a "complete" digraph as having every ordered pair
    (u -> v) present for u != v (self-loops ignored). It measures how many
    arcs are missing on average and how uneven that missingness is across
    vertices, separately for out- and in-degrees.

    For each vertex v:
      - d_out(v) = out-degree excluding any self-loop
      - d_in(v) = in-degree excluding any self-loop
      - x_out(v) = (n - 1) - d_out(v)  (missing out-arcs)
      - x_in(v)  = (n - 1) - d_in(v)   (missing in-arcs)

    Let mu_out/mu_in be the means of x_out/x_in and sigma_out/sigma_in be
    their standard deviations. The score is:

        S = 1 - (mu_out + mu_in) / (2(n-1)) - lambda * (sigma_out + sigma_in) / (2(n-1))

    Higher scores mean closer to a complete, regular digraph. For n >= 2 the
    score is typically in [0, 1], with 1.0 for a complete digraph. lambda
    controls how strongly unevenness is penalized: lambda = 0 ignores
    dispersion and only penalizes missing arcs on average, while larger values
    make outlier vertices reduce the score more aggressively. Runs in O(n + m).
    """
    if not math.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda must be finite and non-negative")

    n = graph.number_of_nodes()
    if n < 2:
        return 0.0

    missing_out: list[float] = []
    missing_in: list[float] = []
    for node in graph.nodes:
        out_degree = graph.out_degree(node)
        in_degree = graph.in_degree(node)
        if graph.has_edge(node, node):
            out_degree -= 1
            in_degree -= 1
        missing_out.append((n - 1) - out_degree)
        missing_in.append((n - 1) - in_degree)

    mu_out = sum(missing_out) / n
    mu_in = sum(missing_in) / n
    sigma_out = math.sqrt(
        sum((value - mu_out) ** 2 for value in missing_out) / n
    )
    sigma_in = math.sqrt(
        sum((value - mu_in) ** 2 for value in missing_in) / n
    )

    denom = 2 * (n - 1)
    return (
        1.0
        - (mu_out + mu_in) / denom
        - lambda_ * (sigma_out + sigma_in) / denom
    )


def is_top_tier(
    graph: nx.DiGraph,
    *,
    score_threshold: float = DEFAULT_TOP_TIER_SCORE_THRESHOLD,
    lambda_: float = DEFAULT_TOP_TIER_LAMBDA,
) -> bool:
    """
    Return True if the graph meets the top-tier score threshold.
    """
    if not math.isfinite(score_threshold):
        raise ValueError("score_threshold must be finite")
    score = compute_top_tier_score(graph, lambda_=lambda_)
    return score >= score_threshold


def _normalize_maxflow_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-")
    if normalized not in MAXFLOW_MODES:
        raise ValueError(
            f"maxflow_mode must be one of {', '.join(MAXFLOW_MODES)}")
    return normalized


def compute_maxflow_scores(
    graph: nx.DiGraph,
    seeds: str | Iterable[str],
    *,
    seed_capacity: float = DEFAULT_MAXFLOW_SEED_CAPACITY,
    mode: str = DEFAULT_MAXFLOW_MODE,
) -> dict[str, float]:
    """
    Compute max-flow scores from a set of trusted orgs.

    Each node has unit capacity (maximum total outflow) except seed nodes,
    which use `seed_capacity`, and the sink node (temporarily infinite). We
    implement this via node-splitting: each node becomes v_in -> v_out with
    its capacity. Graph edges are modeled as v_out -> u_in with infinite
    capacity. A super-source connects to each seed with infinite capacity. For
    every node, compute the maximum flow from the super-source to that node as
    a sink, temporarily giving that sink infinite capacity. Only
    mode="standard" is supported.
    """
    if isinstance(seeds, str):
        seed_list = [seeds]
    else:
        seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    missing = [seed for seed in seed_list if seed not in graph]
    if missing:
        raise ValueError(f"seeds not found in graph: {missing}")

    if seed_capacity <= 0:
        raise ValueError("seed_capacity must be positive")

    nodes = list(graph.nodes)
    if not nodes:
        return {}

    unique_seeds = list(dict.fromkeys(seed_list))
    mode = _normalize_maxflow_mode(mode)
    scores: dict[str, float] = {}
    flow_graph = nx.DiGraph()
    super_source = "__sybil_maxflow_source__"
    flow_graph.add_node(super_source)

    for node in nodes:
        capacity = seed_capacity if node in unique_seeds else 1.0
        flow_graph.add_edge(f"{node}_in", f"{node}_out", capacity=capacity)
    for source, target in graph.edges:
        flow_graph.add_edge(
            f"{source}_out",
            f"{target}_in",
            capacity=float("inf"),
        )
    for seed in unique_seeds:
        flow_graph.add_edge(super_source, f"{seed}_in", capacity=float("inf"))

    for node in nodes:
        sink_edge = (f"{node}_in", f"{node}_out")
        previous_capacity = flow_graph.edges[sink_edge]["capacity"]
        flow_graph.edges[sink_edge]["capacity"] = float("inf")
        source_edge = (super_source, f"{node}_in")
        removed_source_edge = None
        adjusted_sources: dict[tuple[str, str], float] = {}
        if node in unique_seeds and flow_graph.has_edge(*source_edge):
            removed_source_edge = flow_graph.edges[source_edge]["capacity"]
            flow_graph.remove_edge(*source_edge)
            if len(unique_seeds) > 1:
                scale = len(unique_seeds) / (len(unique_seeds) - 1)
                for seed in unique_seeds:
                    if seed == node:
                        continue
                    edge = (f"{seed}_in", f"{seed}_out")
                    if flow_graph.has_edge(*edge):
                        adjusted_sources[edge] = flow_graph.edges[edge]["capacity"]
                        flow_graph.edges[edge]["capacity"] *= scale
        flow_value, _ = nx.maximum_flow(
            flow_graph,
            super_source,
            f"{node}_out",
            capacity="capacity",
        )
        scores[node] = flow_value
        flow_graph.edges[sink_edge]["capacity"] = previous_capacity
        for edge, capacity in adjusted_sources.items():
            if flow_graph.has_edge(*edge):
                flow_graph.edges[edge]["capacity"] = capacity
        if removed_source_edge is not None:
            flow_graph.add_edge(
                source_edge[0],
                source_edge[1],
                capacity=removed_source_edge,
            )

    return scores


def compute_maxflow_scores_sweep(
    graph: nx.DiGraph,
    seeds: str | Iterable[str],
    *,
    seed_capacity: float = DEFAULT_MAXFLOW_SEED_CAPACITY,
    mode: str = DEFAULT_MAXFLOW_MODE,
    sweep_factor: float = 2.0,
    sweep_bimodality_threshold: float = DEFAULT_BIMODALITY_THRESHOLD,
    sweep_max_steps: int = 8,
    sweep_post_threshold_steps: int = DEFAULT_MAXFLOW_SWEEP_POST_THRESHOLD_STEPS,
) -> tuple[dict[str, float], list[float], list[float]]:
    """
    Sweep max-flow seed capacity and return scores with sweep history.
    `mode` is forwarded to `compute_maxflow_scores`.

    Returns (final_scores, capacities, bimodality_coeffs). If
    sweep_post_threshold_steps > 0, the sweep continues for that many
    additional iterations after reaching the bimodality threshold.
    """
    if sweep_factor <= 1.0:
        raise ValueError("sweep_factor must be > 1")
    if sweep_bimodality_threshold <= 0:
        raise ValueError("sweep_bimodality_threshold must be positive")
    if sweep_max_steps < 0:
        raise ValueError("sweep_max_steps must be non-negative")
    if sweep_post_threshold_steps < 0:
        raise ValueError("sweep_post_threshold_steps must be non-negative")

    capacities = [seed_capacity]
    scores = compute_maxflow_scores(
        graph,
        seeds,
        seed_capacity=seed_capacity,
        mode=mode,
    )
    bcs = [compute_bimodality_coefficient(scores.values())]
    current_capacity = seed_capacity

    if sweep_max_steps == 0:
        logging.info(
            "Max-flow sweep disabled; seed_capacity=%.6g",
            seed_capacity,
        )
        return scores, capacities, bcs
    post_threshold_remaining: int | None = None
    if bcs[-1] >= sweep_bimodality_threshold:
        if sweep_post_threshold_steps == 0:
            logging.info(
                "Max-flow sweep stopped after 1 iteration; final seed_capacity=%.6g",
                seed_capacity,
            )
            return scores, capacities, bcs
        post_threshold_remaining = sweep_post_threshold_steps
        logging.info(
            "Max-flow sweep hit bimodality threshold after 1 iteration; "
            "continuing for %d more steps",
            post_threshold_remaining,
        )

    for _ in range(sweep_max_steps):
        current_capacity *= sweep_factor
        scores = compute_maxflow_scores(
            graph,
            seeds,
            seed_capacity=current_capacity,
            mode=mode,
        )
        capacities.append(current_capacity)
        bc = compute_bimodality_coefficient(scores.values())
        bcs.append(bc)
        if post_threshold_remaining is not None:
            post_threshold_remaining -= 1
            if post_threshold_remaining <= 0:
                logging.info(
                    "Max-flow sweep stopped after %d iterations; final seed_capacity=%.6g",
                    len(capacities),
                    current_capacity,
                )
                return scores, capacities, bcs
            continue
        if bc >= sweep_bimodality_threshold:
            if sweep_post_threshold_steps == 0:
                logging.info(
                    "Max-flow sweep stopped after %d iterations; "
                    "final seed_capacity=%.6g",
                    len(capacities),
                    current_capacity,
                )
                return scores, capacities, bcs
            post_threshold_remaining = sweep_post_threshold_steps
            logging.info(
                "Max-flow sweep hit bimodality threshold after %d iterations; "
                "continuing for %d more steps",
                len(capacities),
                post_threshold_remaining,
            )

    logging.info(
        "Max-flow sweep reached max steps (%d); final seed_capacity=%.6g",
        sweep_max_steps,
        current_capacity,
    )
    return scores, capacities, bcs


def _cluster_high_score_nodes(
    scores: dict[str, float],
    *,
    max_iters: int = 100,
) -> set[str]:
    """
    Partition 1D scores into low/high clusters via a simple 2-means iteration.

    The algorithm is k-means in 1D with k=2:
      1) Initialize centers to (min_score, max_score).
      2) Split scores by the midpoint between centers.
      3) Recompute each center as the mean of its assigned cluster.
      4) Repeat until centers stop changing or max_iters is reached.

    The returned set is the cluster with the higher center. If all scores are
    identical or a split degenerates into an empty cluster, the full node set
    is returned.
    """
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if not scores:
        return set()
    values = list(scores.values())
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    min_score = min(values)
    max_score = max(values)
    if min_score == max_score:
        return set(scores.keys())

    low_center = min_score
    high_center = max_score
    high_cluster: set[str] = set()

    for _ in range(max_iters):
        midpoint = (low_center + high_center) / 2.0
        low_cluster: set[str] = set()
        high_cluster = set()
        for node, score in scores.items():
            if score >= midpoint:
                high_cluster.add(node)
            else:
                low_cluster.add(node)
        if not low_cluster or not high_cluster:
            return set(scores.keys())
        new_low = sum(scores[node] for node in low_cluster) / len(low_cluster)
        new_high = sum(scores[node] for node in high_cluster) / len(high_cluster)
        if new_low == low_center and new_high == high_center:
            return high_cluster
        low_center = new_low
        high_center = new_high

    return high_cluster


def extract_non_sybil_cluster_from_scores(
    scores: dict[str, float],
    *,
    kmeans_max_iters: int = 100,
) -> set[str]:
    """
    Cluster nodes by score only and return the higher-score cluster.
    """
    return _cluster_high_score_nodes(scores, max_iters=kmeans_max_iters)


def extract_non_sybil_cluster_maxflow_sweep(
    graph: nx.DiGraph,
    seeds: str | Iterable[str],
    *,
    seed_capacity: float = DEFAULT_MAXFLOW_SEED_CAPACITY,
    mode: str = DEFAULT_MAXFLOW_MODE,
    sweep_factor: float = 2.0,
    sweep_bimodality_threshold: float = DEFAULT_BIMODALITY_THRESHOLD,
    sweep_max_steps: int = 8,
    sweep_post_threshold_steps: int = DEFAULT_MAXFLOW_SWEEP_POST_THRESHOLD_STEPS,
    kmeans_max_iters: int = 100,
) -> tuple[set[str], dict[str, float], list[float], list[float]]:
    """
    Run a max-flow sweep, then cluster nodes by the final scores (scores only),
    and return the higher-score cluster as the non-sybil set.

    Returns (non_sybil_nodes, scores, capacities, bimodality_coeffs).
    """
    scores, capacities, bcs = compute_maxflow_scores_sweep(
        graph,
        seeds,
        seed_capacity=seed_capacity,
        mode=mode,
        sweep_factor=sweep_factor,
        sweep_bimodality_threshold=sweep_bimodality_threshold,
        sweep_max_steps=sweep_max_steps,
        sweep_post_threshold_steps=sweep_post_threshold_steps,
    )
    non_sybil_nodes = _cluster_high_score_nodes(
        scores,
        max_iters=kmeans_max_iters,
    )
    return non_sybil_nodes, scores, capacities, bcs
