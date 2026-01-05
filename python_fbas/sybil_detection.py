from __future__ import annotations

from collections import defaultdict
import logging
from typing import Final, Iterable

try:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    HAS_LP = True
except Exception:
    linprog = None
    coo_matrix = None
    HAS_LP = False

import networkx as nx

DEFAULT_STEPS: Final[int] = 5
DEFAULT_CAPACITY: Final[float] = 1.0
DEFAULT_TRUSTRANK_ALPHA: Final[float] = 0.2
DEFAULT_TRUSTRANK_EPSILON: Final[float] = 1e-8
DEFAULT_MAXFLOW_SEED_CAPACITY: Final[float] = 1.0
DEFAULT_MAXFLOW_MODE: Final[str] = "standard"
MAXFLOW_MODES: Final[tuple[str, ...]] = ("standard", "equal-outflow")
DEFAULT_BIMODALITY_THRESHOLD: Final[float] = 5 / 9


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
    a sink, temporarily giving that sink infinite capacity. Set
    mode="equal-outflow" to enforce equal flow on all outgoing edges of a node
    via a linear program (requires scipy).
    TODO This works well as long as the total output capacity of seed nodes is larger
    than the attacker cut...
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
    if mode == "equal-outflow":
        return _compute_equal_outflow_maxflow_scores(
            graph,
            unique_seeds,
            seed_capacity=seed_capacity,
        )
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


def _compute_equal_outflow_maxflow_scores(
    graph: nx.DiGraph,
    seeds: list[str],
    *,
    seed_capacity: float,
) -> dict[str, float]:
    if not HAS_LP:
        raise RuntimeError(
            "equal-outflow max-flow requires scipy; install python-fbas[lp]")

    nodes = list(graph.nodes)
    if not nodes:
        return {}
    total_seed_capacity = seed_capacity * len(seeds)
    scores: dict[str, float] = {}
    for sink in nodes:
        scores[sink] = _solve_equal_outflow_lp(
            graph,
            nodes,
            seeds,
            seed_capacity=seed_capacity,
            total_seed_capacity=total_seed_capacity,
            sink=sink,
        )
    return scores


def _solve_equal_outflow_lp(
    graph: nx.DiGraph,
    nodes: list[str],
    seeds: list[str],
    *,
    seed_capacity: float,
    total_seed_capacity: float,
    sink: str,
) -> float:
    import numpy as np

    if total_seed_capacity <= 0:
        raise ValueError("seed_capacity must be positive")

    super_source = "__sybil_maxflow_source__"
    big_m = total_seed_capacity
    node_caps = {
        node: seed_capacity if node in seeds else 1.0
        for node in nodes
    }

    if sink in seeds and len(seeds) > 1:
        scale = len(seeds) / (len(seeds) - 1)
        for seed in seeds:
            if seed != sink:
                node_caps[seed] *= scale
    node_caps[sink] = big_m

    edges: list[tuple[str, str, float, str, str | None]] = []
    incoming: dict[str, list[int]] = defaultdict(list)
    outgoing: dict[str, list[int]] = defaultdict(list)
    orig_out_edges: dict[str, list[int]] = defaultdict(list)

    def add_edge(
        source: str,
        target: str,
        capacity: float,
        kind: str,
        origin: str | None = None,
    ) -> int:
        edge_idx = len(edges)
        edges.append((source, target, capacity, kind, origin))
        outgoing[source].append(edge_idx)
        incoming[target].append(edge_idx)
        if kind == "orig" and origin is not None:
            orig_out_edges[origin].append(edge_idx)
        return edge_idx

    sink_edge_idx = None
    for node in nodes:
        edge_idx = add_edge(
            f"{node}_in",
            f"{node}_out",
            node_caps[node],
            "node",
        )
        if node == sink:
            sink_edge_idx = edge_idx

    for source, target in graph.edges:
        if source == sink:
            continue
        add_edge(
            f"{source}_out",
            f"{target}_in",
            big_m,
            "orig",
            origin=source,
        )

    for seed in seeds:
        if seed == sink:
            continue
        add_edge(
            super_source,
            f"{seed}_in",
            big_m,
            "source",
        )

    if sink_edge_idx is None:
        raise RuntimeError("Missing sink edge in max-flow LP")

    outflow_var_index: dict[str, int] = {}
    next_var = len(edges)
    for node, edge_indices in orig_out_edges.items():
        if node == sink or len(edge_indices) < 2:
            continue
        outflow_var_index[node] = next_var
        next_var += 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_eq: list[float] = []
    row = 0

    split_nodes = [super_source]
    for node in nodes:
        split_nodes.append(f"{node}_in")
        split_nodes.append(f"{node}_out")

    sink_out = f"{sink}_out"
    for node in split_nodes:
        if node in {super_source, sink_out}:
            continue
        for edge_idx in incoming.get(node, []):
            rows.append(row)
            cols.append(edge_idx)
            data.append(1.0)
        for edge_idx in outgoing.get(node, []):
            rows.append(row)
            cols.append(edge_idx)
            data.append(-1.0)
        b_eq.append(0.0)
        row += 1

    # Enforce equal flow on all outgoing edges of a node.
    for node, out_idx in outflow_var_index.items():
        for edge_idx in orig_out_edges[node]:
            rows.append(row)
            cols.append(edge_idx)
            data.append(1.0)
            rows.append(row)
            cols.append(out_idx)
            data.append(-1.0)
            b_eq.append(0.0)
            row += 1

    num_vars = len(edges) + len(outflow_var_index)
    a_eq = coo_matrix((data, (rows, cols)), shape=(row, num_vars))
    b_eq_array = np.array(b_eq)

    c = np.zeros(num_vars)
    c[sink_edge_idx] = -1.0

    bounds = [(0.0, cap) for _src, _dst, cap, _kind, _origin in edges]
    bounds.extend((0.0, None) for _ in outflow_var_index)

    result = linprog(
        c,
        A_eq=a_eq,
        b_eq=b_eq_array,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"Equal-outflow max-flow LP failed for sink {sink}: {result.message}")
    return max(0.0, float(result.x[sink_edge_idx]))


def compute_maxflow_scores_sweep(
    graph: nx.DiGraph,
    seeds: str | Iterable[str],
    *,
    seed_capacity: float = DEFAULT_MAXFLOW_SEED_CAPACITY,
    mode: str = DEFAULT_MAXFLOW_MODE,
    sweep_factor: float = 2.0,
    sweep_bimodality_threshold: float = DEFAULT_BIMODALITY_THRESHOLD,
    sweep_max_steps: int = 8,
) -> tuple[dict[str, float], list[float], list[float]]:
    """
    Sweep max-flow seed capacity and return scores with sweep history.
    `mode` is forwarded to `compute_maxflow_scores`.

    Returns (final_scores, capacities, bimodality_coeffs).
    """
    if sweep_factor <= 1.0:
        raise ValueError("sweep_factor must be > 1")
    if sweep_bimodality_threshold <= 0:
        raise ValueError("sweep_bimodality_threshold must be positive")
    if sweep_max_steps < 0:
        raise ValueError("sweep_max_steps must be non-negative")

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
    if bcs[-1] >= sweep_bimodality_threshold:
        logging.info(
            "Max-flow sweep stopped after 1 iteration; final seed_capacity=%.6g",
            seed_capacity,
        )
        return scores, capacities, bcs

    for step in range(sweep_max_steps):
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
        if bc >= sweep_bimodality_threshold:
            logging.info(
                "Max-flow sweep stopped after %d iterations; final seed_capacity=%.6g",
                step + 2,
                current_capacity,
            )
            return scores, capacities, bcs

    logging.info(
        "Max-flow sweep reached max steps (%d); final seed_capacity=%.6g",
        sweep_max_steps,
        current_capacity,
    )
    return scores, capacities, bcs
