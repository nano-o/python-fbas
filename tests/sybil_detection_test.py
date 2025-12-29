import random
import statistics

import networkx as nx

from python_fbas.fbas_generator import (
    SybilAttackConfig,
    gen_random_sybil_attack_org_graph,
)
from python_fbas.sybil_detection import compute_trust_scores


def test_trust_scores_simple_chain():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
    ])
    scores = compute_trust_scores(graph, ["seed"], steps=2, capacity=1.0)
    assert scores["seed"] == 1.0
    assert scores["a"] == 1.0
    assert scores["b"] == 1.0


def test_trust_scores_capacity_limit():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("seed", "b"),
    ])
    scores = compute_trust_scores(graph, ["seed"], steps=1, capacity=0.5)
    assert scores["seed"] == 1.5
    assert scores["a"] == 0.25
    assert scores["b"] == 0.25


def test_sybil_attack_scores_honest_higher():
    rng = random.Random(0)
    config = SybilAttackConfig(
        original_edge_probability=0.7,
        sybil_sybil_edge_probability=0.7,
        attacker_to_sybil_edge_probability=0.9,
        connect_attacker_to_attacker=False,
        connect_attacker_to_honest=False,
        connect_sybil_to_honest=False,
        connect_sybil_to_attacker=False,
        max_attempts=100,
    )
    graph = gen_random_sybil_attack_org_graph(
        num_orgs=10,
        num_sybil_orgs=6,
        config=config,
        rng=rng,
    )
    roles = nx.get_node_attributes(graph, "role")
    honest_nodes = [n for n, role in roles.items() if role == "honest"]
    sybil_nodes = [n for n, role in roles.items() if role == "sybil"]
    seeds = honest_nodes[:2]

    scores = compute_trust_scores(graph, seeds, steps=5, capacity=1.0)
    honest_mean = statistics.mean(scores[n] for n in honest_nodes)
    sybil_mean = statistics.mean(scores[n] for n in sybil_nodes)
    assert honest_mean > sybil_mean
