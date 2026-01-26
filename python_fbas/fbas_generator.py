from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx

from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import (
    find_disjoint_quorums,
    find_min_cardinality_min_quorum,
    random_quorum,
)
from python_fbas.org_graph import org_graph_to_fbas, org_graph_to_org_level_fbas


@dataclass(frozen=True)
class SybilAttackConfig:
    original_edge_probability: float = 0.66
    sybil_sybil_edge_probability: float = 0.66
    sybil2_sybil2_edge_probability: float = 0.66
    attacker_to_sybil_edge_probability: float = 0.66
    attacker_to_attacker_edge_probability: float = 0.66
    attacker_to_honest_edge_probability: float = 0.66
    sybil_to_honest_edge_probability: float = 0.66
    sybil_to_attacker_edge_probability: float = 0.66
    sybil_to_sybil_bridge_edge_probability: float = 0.66
    sybil_bridge_to_sybil2_edge_probability: float = 0.66
    sybil_bridge_to_sybil_bridge_edge_probability: float = 0.66
    sybil2_to_honest_edge_probability: float = 0.66
    sybil2_to_attacker_edge_probability: float = 0.66
    sybil2_to_sybil1_edge_probability: float = 0.66
    sybil2_to_sybil_bridge_edge_probability: float = 0.66
    sybil1_to_sybil2_edge_probability: float = 0.66
    connect_attacker_to_attacker: bool = False
    connect_attacker_to_honest: bool = False
    connect_sybil_to_honest: bool = False
    connect_sybil_to_attacker: bool = False
    connect_sybil_bridge_to_sybil_bridge: bool = False
    connect_sybil2_to_honest: bool = False
    connect_sybil2_to_attacker: bool = False
    connect_sybil2_to_sybil1: bool = False
    connect_sybil2_to_sybil_bridge: bool = False
    connect_sybil1_to_sybil2: bool = False
    max_attempts: int = 100


def gen_random_top_tier_org_graph(
    num_orgs: int,
    *,
    edge_probability: float = 0.66,
    max_threshold_ratio: float = 0.85,
    rng: random.Random | None = None,
    org_prefix: str = "org",
) -> nx.DiGraph:
    """
    Generate a random top-tier org graph.

    The result is a directed graph whose vertices are organizations labeled
    with a "threshold" attribute between half and max_threshold_ratio of their
    out-degree. Each potential edge is added with the given probability, with a
    fallback to ensure at least one outgoing edge per org.
    """
    if num_orgs < 2:
        raise ValueError("num_orgs must be at least 2 to form a top-tier org graph")

    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")

    if not 0 < max_threshold_ratio <= 1:
        raise ValueError("max_threshold_ratio must be in (0, 1]")

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
        min_threshold = (len(targets) + 1) // 2
        max_threshold = max(
            min_threshold,
            math.floor(len(targets) * max_threshold_ratio),
        )
        graph.nodes[org]["threshold"] = rng.randint(
            min_threshold,
            max_threshold,
        )

    return graph


def gen_random_top_tier_org_fbas(
    num_orgs: int,
    *,
    edge_probability: float = 0.66,
    max_threshold_ratio: float = 0.85,
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
            max_threshold_ratio=max_threshold_ratio,
            rng=rng,
        )
        # Use fake FBAS for quorum check (1 validator per org)
        fake_fbas = org_graph_to_org_level_fbas(top_tier)
        if find_disjoint_quorums(fake_fbas) is None:
            # Return the real org-structured FBAS
            return org_graph_to_fbas(top_tier)

    raise ValueError(
        "Failed to generate an FBAS with intersecting quorums after "
        f"{max_attempts} attempts")


def gen_random_sybil_attack_org_graph(
    num_orgs: int,
    num_sybil_orgs: int,
    *,
    num_sybil_clusters: int = 1,
    num_sybil_orgs_2: int = 0,
    num_sybil_bridge_orgs: int = 0,
    quorum_selection: str = "random",
    max_threshold_ratio: float = 0.85,
    config: SybilAttackConfig | None = None,
    rng: random.Random | None = None,
) -> nx.DiGraph:
    """
    Generate a top-tier org graph that simulates a Sybil attack.

    This uses rejection sampling and returns the first construction that meets
    the quorum constraints.

    Procedure (per attempt):
    1) Generate an "original" top-tier org graph and reject it if its FBAS has
       disjoint quorums.
    2) Select a quorum in the original FBAS (random or minimum-cardinality).
       Nodes in the quorum are "honest"; the complement are "attackers". Reject
       if there are no attackers.
    3) Remove attacker -> honest edges from the original graph.
    4) Generate a Sybil cluster (sybil-*) and optionally a second cluster
       (sybil2-*) plus bridge nodes (sybil-bridge-*). For two clusters, add at
       least one edge from each sybil1 to a bridge and from each bridge to a
       sybil2; optional sybil1->sybil2 and bridge->bridge edges are added based
       on the config probabilities.
    5) Merge nodes/edges and annotate roles: honest/attacker/sybil/
       sybil_sybil_bridge, plus sybil_cluster for sybil nodes.
    6) Add edges from each attacker to sybil nodes (at least one per attacker).
       Optional attacker->attacker and attacker->honest edges are added per
       config.
    7) Optionally add edges from sybils to honest/attackers, and from sybil2
       nodes to honest/attackers/sybil1/bridges.
    8) Recompute thresholds for attackers and for any sybil/bridge nodes that
       gained new outgoing edges; thresholds are uniform in
       [ceil(out_degree / 2), floor(out_degree * max_threshold_ratio)] (with the
       upper bound clamped to be at least the lower bound).
    """
    if num_orgs < 2:
        raise ValueError("num_orgs must be at least 2")

    if num_sybil_clusters not in {1, 2}:
        raise ValueError("num_sybil_clusters must be 1 or 2")

    if num_sybil_orgs < 2:
        raise ValueError("num_sybil_orgs must be at least 2")

    if quorum_selection not in {"random", "min"}:
        raise ValueError("quorum_selection must be 'random' or 'min'")

    if num_sybil_clusters == 2:
        if num_sybil_orgs_2 < 2:
            raise ValueError("num_sybil_orgs_2 must be at least 2")
        if num_sybil_bridge_orgs < 1:
            raise ValueError("num_sybil_bridge_orgs must be at least 1")

    if config is None:
        config = SybilAttackConfig()

    for name, value in (
        ("original_edge_probability", config.original_edge_probability),
        ("sybil_sybil_edge_probability", config.sybil_sybil_edge_probability),
        ("sybil2_sybil2_edge_probability", config.sybil2_sybil2_edge_probability),
        ("attacker_to_sybil_edge_probability",
         config.attacker_to_sybil_edge_probability),
        ("attacker_to_attacker_edge_probability",
         config.attacker_to_attacker_edge_probability),
        ("attacker_to_honest_edge_probability",
         config.attacker_to_honest_edge_probability),
        ("sybil_to_honest_edge_probability", config.sybil_to_honest_edge_probability),
        ("sybil_to_attacker_edge_probability",
         config.sybil_to_attacker_edge_probability),
        ("sybil_to_sybil_bridge_edge_probability",
         config.sybil_to_sybil_bridge_edge_probability),
        ("sybil_bridge_to_sybil2_edge_probability",
         config.sybil_bridge_to_sybil2_edge_probability),
        ("sybil_bridge_to_sybil_bridge_edge_probability",
         config.sybil_bridge_to_sybil_bridge_edge_probability),
        ("sybil2_to_honest_edge_probability",
         config.sybil2_to_honest_edge_probability),
        ("sybil2_to_attacker_edge_probability",
         config.sybil2_to_attacker_edge_probability),
        ("sybil2_to_sybil1_edge_probability",
         config.sybil2_to_sybil1_edge_probability),
        ("sybil2_to_sybil_bridge_edge_probability",
         config.sybil2_to_sybil_bridge_edge_probability),
        ("sybil1_to_sybil2_edge_probability",
         config.sybil1_to_sybil2_edge_probability),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if rng is None:
        rng = random.Random()

    def random_threshold(out_degree: int) -> int:
        min_threshold = (out_degree + 1) // 2
        max_threshold = max(
            min_threshold,
            math.floor(out_degree * max_threshold_ratio),
        )
        return rng.randint(min_threshold, max_threshold)

    for _ in range(config.max_attempts):
        original = gen_random_top_tier_org_graph(
            num_orgs,
            edge_probability=config.original_edge_probability,
            max_threshold_ratio=max_threshold_ratio,
            rng=rng,
            org_prefix="org",
        )
        # Use fake FBAS for quorum check (1 validator per org)
        original_fbas = org_graph_to_org_level_fbas(original)
        if find_disjoint_quorums(original_fbas) is not None:
            continue
        sybil = gen_random_top_tier_org_graph(
            num_sybil_orgs,
            edge_probability=config.sybil_sybil_edge_probability,
            max_threshold_ratio=max_threshold_ratio,
            rng=rng,
            org_prefix="sybil",
        )
        sybil2 = None
        if num_sybil_clusters == 2:
            sybil2 = gen_random_top_tier_org_graph(
                num_sybil_orgs_2,
                edge_probability=config.sybil2_sybil2_edge_probability,
                max_threshold_ratio=max_threshold_ratio,
                rng=rng,
                org_prefix="sybil2",
            )
        if quorum_selection == "min":
            quorum = find_min_cardinality_min_quorum(
                original_fbas,
                project_on_scc=False)
            if not quorum:
                continue
        else:
            quorum = random_quorum(
                original_fbas,
                seed=rng.randrange(1 << 63),
            )
            if quorum is None:
                continue

        quorum_set = set(quorum)
        honest_orgs = [node for node in original.nodes if node in quorum_set]
        attackers = [node for node in original.nodes if node not in quorum_set]
        if not attackers:
            continue
        attackers_set = set(attackers)

        for attacker in attackers:
            for target in list(original.successors(attacker)):
                if target in quorum_set:
                    original.remove_edge(attacker, target)

        combined = nx.DiGraph()
        combined.add_nodes_from(original.nodes(data=True))
        combined.add_edges_from(original.edges())
        combined.add_nodes_from(sybil.nodes(data=True))
        combined.add_edges_from(sybil.edges())
        if sybil2 is not None:
            combined.add_nodes_from(sybil2.nodes(data=True))
            combined.add_edges_from(sybil2.edges())

        sybil_nodes = list(sybil.nodes)
        sybil2_nodes = list(sybil2.nodes) if sybil2 is not None else []
        bridge_nodes = []
        if num_sybil_clusters == 2:
            bridge_nodes = [
                f"sybil-bridge-{i}" for i in range(num_sybil_bridge_orgs)
            ]
            combined.add_nodes_from(bridge_nodes)
        for org in original.nodes:
            role = "attacker" if org in attackers_set else "honest"
            combined.nodes[org]["role"] = role
        for sybil_org in sybil_nodes:
            combined.nodes[sybil_org]["role"] = "sybil"
            combined.nodes[sybil_org]["sybil_cluster"] = 1
        for sybil_org in sybil2_nodes:
            combined.nodes[sybil_org]["role"] = "sybil"
            combined.nodes[sybil_org]["sybil_cluster"] = 2
        for bridge_org in bridge_nodes:
            combined.nodes[bridge_org]["role"] = "sybil_sybil_bridge"
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
            combined.nodes[attacker]["threshold"] = random_threshold(out_degree)

        if config.connect_sybil_to_honest or config.connect_sybil_to_attacker:
            honest_targets = honest_orgs if config.connect_sybil_to_honest else []
            attacker_targets = attackers if config.connect_sybil_to_attacker else []
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
                    if num_sybil_clusters == 1:
                        out_degree = combined.out_degree(sybil_org)
                        combined.nodes[sybil_org]["threshold"] = random_threshold(
                            out_degree
                        )

        if num_sybil_clusters == 2:
            for sybil_org in sybil_nodes:
                bridge_targets = [
                    target for target in bridge_nodes
                    if rng.random() < config.sybil_to_sybil_bridge_edge_probability
                ]
                if not bridge_targets:
                    bridge_targets = [rng.choice(bridge_nodes)]
                combined.add_edges_from(
                    (sybil_org, target) for target in bridge_targets
                )
                if config.connect_sybil1_to_sybil2:
                    for target in sybil2_nodes:
                        if rng.random() < config.sybil1_to_sybil2_edge_probability:
                            combined.add_edge(sybil_org, target)

            if config.connect_sybil_bridge_to_sybil_bridge:
                for bridge_org in bridge_nodes:
                    for other_bridge in bridge_nodes:
                        if bridge_org != other_bridge:
                            if rng.random() < (
                                config.sybil_bridge_to_sybil_bridge_edge_probability
                            ):
                                combined.add_edge(bridge_org, other_bridge)

            for bridge_org in bridge_nodes:
                sybil2_targets = [
                    target for target in sybil2_nodes
                    if rng.random() < config.sybil_bridge_to_sybil2_edge_probability
                ]
                if not sybil2_targets:
                    sybil2_targets = [rng.choice(sybil2_nodes)]
                combined.add_edges_from(
                    (bridge_org, target) for target in sybil2_targets
                )
                out_degree = combined.out_degree(bridge_org)
                if out_degree == 0:
                    fallback = rng.choice(sybil2_nodes)
                    combined.add_edge(bridge_org, fallback)
                    out_degree = 1
                combined.nodes[bridge_org]["threshold"] = random_threshold(out_degree)

            if (
                config.connect_sybil2_to_honest
                or config.connect_sybil2_to_attacker
                or config.connect_sybil2_to_sybil1
                or config.connect_sybil2_to_sybil_bridge
            ):
                sybil2_targets_honest = (
                    honest_orgs if config.connect_sybil2_to_honest else []
                )
                sybil2_targets_attackers = (
                    attackers if config.connect_sybil2_to_attacker else []
                )
                sybil2_targets_sybil1 = (
                    sybil_nodes if config.connect_sybil2_to_sybil1 else []
                )
                sybil2_targets_bridge = (
                    bridge_nodes if config.connect_sybil2_to_sybil_bridge else []
                )
                for sybil2_org in sybil2_nodes:
                    for target in (
                        sybil2_targets_honest
                        + sybil2_targets_attackers
                        + sybil2_targets_sybil1
                        + sybil2_targets_bridge
                    ):
                        if target in sybil2_targets_honest:
                            edge_probability = (
                                config.sybil2_to_honest_edge_probability
                            )
                        elif target in sybil2_targets_attackers:
                            edge_probability = (
                                config.sybil2_to_attacker_edge_probability
                            )
                        elif target in sybil2_targets_sybil1:
                            edge_probability = (
                                config.sybil2_to_sybil1_edge_probability
                            )
                        else:
                            edge_probability = (
                                config.sybil2_to_sybil_bridge_edge_probability
                            )
                        if rng.random() < edge_probability:
                            combined.add_edge(sybil2_org, target)
                    out_degree = combined.out_degree(sybil2_org)
                    combined.nodes[sybil2_org]["threshold"] = random_threshold(
                        out_degree
                    )

            for sybil_org in sybil_nodes:
                out_degree = combined.out_degree(sybil_org)
                combined.nodes[sybil_org]["threshold"] = random_threshold(out_degree)

        return combined

    raise ValueError(
        "Failed to generate a Sybil-attack FBAS after "
        f"{config.max_attempts} attempts")


def gen_random_sybil_attack_fbas(
    num_orgs: int,
    num_sybil_orgs: int,
    *,
    num_sybil_clusters: int = 1,
    num_sybil_orgs_2: int = 0,
    num_sybil_bridge_orgs: int = 0,
    quorum_selection: str = "random",
    max_threshold_ratio: float = 0.85,
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
        num_sybil_clusters=num_sybil_clusters,
        num_sybil_orgs_2=num_sybil_orgs_2,
        num_sybil_bridge_orgs=num_sybil_bridge_orgs,
        quorum_selection=quorum_selection,
        max_threshold_ratio=max_threshold_ratio,
        config=config,
        rng=rng,
    )
    return org_graph_to_fbas(graph)
