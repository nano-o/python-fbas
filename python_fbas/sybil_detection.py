from __future__ import annotations

from collections import defaultdict
from typing import Final

import networkx as nx

DEFAULT_STEPS: Final[int] = 5
DEFAULT_CAPACITY: Final[float] = 1.0


def compute_trust_scores(
    graph: nx.DiGraph,
    seed: str,
    *,
    capacity: float = DEFAULT_CAPACITY,
    steps: int = DEFAULT_STEPS,
) -> dict[str, float]:
    """
    Compute capacity-limited trust scores from a seed org.

    Trust mass starts at the seed (1.0) and is propagated along outgoing edges
    for a fixed number of steps. Each node may send at most `capacity` total
    mass per step, split evenly among its outgoing neighbors. Any remaining
    mass stays at the node. Scores are the cumulative mass received over all
    steps, including the initial seed mass.

    This design intentionally uses a fixed number of steps with cumulative
    scoring. A fix-point is only meaningful for steady-state mass, which would
    require decay/restart and tends to leak more trust across small cuts.
    Bounded walks keep trust local, which is the desired sybil-resistance
    behavior.
    """
    if seed not in graph:
        raise ValueError(f"seed '{seed}' not found in graph")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    scores: dict[str, float] = {node: 0.0 for node in graph.nodes}
    mass: dict[str, float] = {seed: 1.0}
    scores[seed] = 1.0

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
