from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import find_disjoint_quorums, random_quorum


@dataclass(frozen=True)
class SybilAttackConfig:
    original_edge_probability: float = 0.66
    sybil_sybil_edge_probability: float = 0.66
    attacker_to_sybil_edge_probability: float = 0.66
    attacker_to_attacker_edge_probability: float = 0.66
    attacker_to_honest_edge_probability: float = 0.66
    sybil_to_honest_edge_probability: float = 0.66
    sybil_to_attacker_edge_probability: float = 0.66
    connect_attacker_to_attacker: bool = False
    connect_attacker_to_honest: bool = False
    connect_sybil_to_honest: bool = False
    connect_sybil_to_attacker: bool = False
    max_attempts: int = 100


def gen_random_top_tier_org_graph(
    num_orgs: int,
    *,
    edge_probability: float = 0.66,
    rng: random.Random | None = None,
    org_prefix: str = "org",
) -> nx.DiGraph:
    """
    Generate a random top-tier org graph.

    The result is a directed graph whose vertices are organizations labeled
    with a "threshold" attribute between 1 and their out-degree. Each
    potential edge is added with the given probability, with a fallback to
    ensure at least one outgoing edge per org.
    """
    if num_orgs < 2:
        raise ValueError("num_orgs must be at least 2 to form a top-tier org graph")

    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")

    if rng is None:
        rng = random.Random()

    org_ids = [f"{org_prefix}-{i}" for i in range(num_orgs)]
    graph = nx.DiGraph()
    graph.add_nodes_from(org_ids)

    for org in org_ids:
        candidates = [candidate for candidate in org_ids if candidate != org]
        targets = [candidate for candidate in candidates
                   if rng.random() < edge_probability]
        if not targets:
            targets = [rng.choice(candidates)]
        graph.add_edges_from((org, target) for target in targets)
        graph.nodes[org]["threshold"] = rng.randint(1, len(targets))

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


def gen_random_top_tier_org_fbas(
    num_orgs: int,
    *,
    edge_probability: float = 0.66,
    rng: random.Random | None = None,
    max_attempts: int = 100,
) -> FBASGraph:
    """
    Generate a random top-tier org FBASGraph with intersecting quorums.

    Uses rejection sampling based on find_disjoint_quorums.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if rng is None:
        rng = random.Random()

    for _ in range(max_attempts):
        top_tier = gen_random_top_tier_org_graph(
            num_orgs,
            edge_probability=edge_probability,
            rng=rng,
        )
        fbas = top_tier_org_graph_to_fbas_graph(top_tier)
        if find_disjoint_quorums(fbas) is None:
            return fbas

    raise ValueError(
        "Failed to generate an FBAS with intersecting quorums after "
        f"{max_attempts} attempts")


def gen_random_sybil_attack_org_graph(
    num_orgs: int,
    num_sybil_orgs: int,
    *,
    config: SybilAttackConfig | None = None,
    rng: random.Random | None = None,
) -> nx.DiGraph:
    """
    Generate a top-tier org graph that simulates a Sybil attack.

    The procedure:
    - Build an original top-tier org graph and a separate Sybil org graph.
    - Sample a random quorum in the original graph; its complement are attackers.
    - Remove attacker edges to non-attacker orgs.
    - For each attacker, add edges to Sybil orgs based on probability and
      randomize its threshold.
    - Optionally add probabilistic edges between attackers, to honest orgs,
      and from Sybil orgs to honest orgs and/or attackers.
    """
    if num_orgs < 2:
        raise ValueError("num_orgs must be at least 2")

    if num_sybil_orgs < 2:
        raise ValueError("num_sybil_orgs must be at least 2")

    if config is None:
        config = SybilAttackConfig()

    for name, value in (
        ("original_edge_probability", config.original_edge_probability),
        ("sybil_sybil_edge_probability", config.sybil_sybil_edge_probability),
        ("attacker_to_sybil_edge_probability",
         config.attacker_to_sybil_edge_probability),
        ("attacker_to_attacker_edge_probability",
         config.attacker_to_attacker_edge_probability),
        ("attacker_to_honest_edge_probability",
         config.attacker_to_honest_edge_probability),
        ("sybil_to_honest_edge_probability", config.sybil_to_honest_edge_probability),
        ("sybil_to_attacker_edge_probability",
         config.sybil_to_attacker_edge_probability),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if rng is None:
        rng = random.Random()

    for _ in range(config.max_attempts):
        original = gen_random_top_tier_org_graph(
            num_orgs,
            edge_probability=config.original_edge_probability,
            rng=rng,
            org_prefix="org",
        )
        original_fbas = top_tier_org_graph_to_fbas_graph(original)
        if find_disjoint_quorums(original_fbas) is not None:
            continue
        sybil = gen_random_top_tier_org_graph(
            num_sybil_orgs,
            edge_probability=config.sybil_sybil_edge_probability,
            rng=rng,
            org_prefix="sybil",
        )
        quorum = random_quorum(
            original_fbas,
            seed=rng.randrange(1 << 63),
        )
        if quorum is None:
            continue

        attackers = set(original.nodes) - set(quorum)
        if not attackers:
            continue

        for attacker in attackers:
            for target in list(original.successors(attacker)):
                if target in quorum:
                    original.remove_edge(attacker, target)

        combined = nx.DiGraph()
        combined.add_nodes_from(original.nodes(data=True))
        combined.add_edges_from(original.edges())
        combined.add_nodes_from(sybil.nodes(data=True))
        combined.add_edges_from(sybil.edges())

        sybil_nodes = list(sybil.nodes)
        honest_orgs = list(quorum)
        for org in original.nodes:
            role = "attacker" if org in attackers else "honest"
            combined.nodes[org]["role"] = role
        for sybil_org in sybil_nodes:
            combined.nodes[sybil_org]["role"] = "sybil"
        for attacker in attackers:
            sybil_targets = [target for target in sybil_nodes
                             if rng.random()
                             < config.attacker_to_sybil_edge_probability]
            if not sybil_targets:
                sybil_targets = [rng.choice(sybil_nodes)]
            combined.add_edges_from((attacker, target) for target in sybil_targets)

        if config.connect_attacker_to_attacker:
            for other_attacker in attackers:
                if other_attacker != attacker:
                    if rng.random() < config.attacker_to_attacker_edge_probability:
                        combined.add_edge(attacker, other_attacker)

            if config.connect_attacker_to_honest and honest_orgs:
                for target in honest_orgs:
                    if rng.random() < config.attacker_to_honest_edge_probability:
                        combined.add_edge(attacker, target)

        for attacker in attackers:
            out_degree = combined.out_degree(attacker)
            if out_degree == 0:
                fallback = rng.choice(sybil_nodes)
                combined.add_edge(attacker, fallback)
                out_degree = 1
            combined.nodes[attacker]["threshold"] = rng.randint(1, out_degree)

        if config.connect_sybil_to_honest or config.connect_sybil_to_attacker:
            honest_targets = (
                list(quorum) if config.connect_sybil_to_honest else []
            )
            attacker_targets = (
                list(attackers) if config.connect_sybil_to_attacker else []
            )
            original_targets = honest_targets + attacker_targets
            if original_targets:
                for sybil_org in sybil_nodes:
                    for target in original_targets:
                        edge_probability = (
                            config.sybil_to_honest_edge_probability
                            if target in honest_targets
                            else config.sybil_to_attacker_edge_probability
                        )
                        if rng.random() < edge_probability:
                            combined.add_edge(sybil_org, target)
                    out_degree = combined.out_degree(sybil_org)
                    combined.nodes[sybil_org]["threshold"] = rng.randint(1, out_degree)

        return combined

    raise ValueError(
        "Failed to generate a Sybil-attack FBAS after "
        f"{max_attempts} attempts")


def gen_random_sybil_attack_fbas(
    num_orgs: int,
    num_sybil_orgs: int,
    *,
    config: SybilAttackConfig | None = None,
    rng: random.Random | None = None,
) -> FBASGraph:
    """
    Generate an FBASGraph that simulates a Sybil attack.

    See gen_random_sybil_attack_org_graph for the generation procedure.
    """
    graph = gen_random_sybil_attack_org_graph(
        num_orgs,
        num_sybil_orgs,
        config=config,
        rng=rng,
    )
    return top_tier_org_graph_to_fbas_graph(graph)
