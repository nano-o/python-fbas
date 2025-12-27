from __future__ import annotations

import random

import networkx as nx

from python_fbas.fbas_graph import FBASGraph


def gen_random_top_tier_org_graph(
    num_orgs: int,
    *,
    min_out_degree: int = 1,
    max_out_degree: int | None = None,
    rng: random.Random | None = None,
) -> nx.DiGraph:
    """
    Generate a random top-tier org graph.

    The result is a directed graph whose vertices are organizations labeled
    with a "threshold" attribute between 1 and their out-degree.
    """
    if num_orgs < 2:
        raise ValueError("num_orgs must be at least 2 to form a top-tier org graph")

    if max_out_degree is None:
        max_out_degree = num_orgs - 1

    if min_out_degree < 1:
        raise ValueError("min_out_degree must be at least 1")

    if max_out_degree > num_orgs - 1:
        raise ValueError("max_out_degree cannot exceed num_orgs - 1")

    if min_out_degree > max_out_degree:
        raise ValueError("min_out_degree cannot exceed max_out_degree")

    if rng is None:
        rng = random.Random()

    org_ids = [f"org-{i}" for i in range(num_orgs)]
    graph = nx.DiGraph()
    graph.add_nodes_from(org_ids)

    for org in org_ids:
        out_degree = rng.randint(min_out_degree, max_out_degree)
        candidates = [candidate for candidate in org_ids if candidate != org]
        targets = rng.sample(candidates, out_degree)
        graph.add_edges_from((org, target) for target in targets)
        graph.nodes[org]["threshold"] = rng.randint(1, out_degree)

    return graph


def top_tier_org_graph_to_fbas_graph(top_tier: nx.DiGraph) -> FBASGraph:
    """
    Convert a top-tier org graph into an FBASGraph using orgs as validators.

    Each org vertex becomes a validator whose quorum set is formed by its
    outgoing neighbors and the vertex "threshold" attribute.
    """
    fbas = FBASGraph()

    for org in top_tier.nodes:
        fbas.add_validator(org)

    for org in top_tier.nodes:
        out_orgs = list(top_tier.successors(org))
        if not out_orgs:
            raise ValueError(
                f"Top-tier org graph node {org} has no outgoing edges; cannot build qset")

        threshold = top_tier.nodes[org].get("threshold")
        if threshold is None:
            raise ValueError(
                f"Top-tier org graph node {org} is missing a 'threshold' attribute")

        if not isinstance(threshold, int):
            raise ValueError(
                f"Top-tier org graph node {org} has non-integer threshold {threshold}")

        if not 1 <= threshold <= len(out_orgs):
            raise ValueError(
                f"Top-tier org graph node {org} has threshold {threshold} outside "
                f"[1, {len(out_orgs)}]")

        qset_id = fbas.add_qset(threshold, out_orgs)
        fbas.update_validator(org, qset=qset_id)

    return fbas
