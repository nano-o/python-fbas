import random
import statistics

import networkx as nx
import pytest

import python_fbas.sybil_detection as sybil_detection
from python_fbas.fbas_generator import (
    SybilAttackConfig,
    gen_random_sybil_attack_org_graph,
    gen_random_top_tier_org_graph,
)
from python_fbas.org_graph import fbas_to_org_graph
from python_fbas.solver import HAS_QBF
from python_fbas.sybil_detection import (
    estimate_seed_reachability_monte_carlo,
    compute_maxflow_scores,
    compute_maxflow_scores_sweep,
    compute_trust_scores,
    compute_trustrank_scores,
    extract_non_sybil_cluster_maxflow_sweep,
    is_top_tier,
)
from test_utils import load_fbas_from_test_file


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


def test_sybil_attack_two_clusters_smoke():
    rng = random.Random(1)
    config = SybilAttackConfig(
        original_edge_probability=0.7,
        sybil_sybil_edge_probability=0.7,
        sybil2_sybil2_edge_probability=0.7,
        attacker_to_sybil_edge_probability=0.9,
        sybil_to_sybil_bridge_edge_probability=0.9,
        sybil_bridge_to_sybil2_edge_probability=0.9,
        connect_attacker_to_attacker=False,
        connect_attacker_to_honest=False,
        connect_sybil_to_honest=False,
        connect_sybil_to_attacker=False,
        connect_sybil_bridge_to_sybil_bridge=False,
        connect_sybil2_to_honest=False,
        connect_sybil2_to_attacker=False,
        connect_sybil2_to_sybil1=False,
        connect_sybil2_to_sybil_bridge=False,
        connect_sybil1_to_sybil2=False,
        max_attempts=200,
    )
    graph = gen_random_sybil_attack_org_graph(
        num_orgs=10,
        num_sybil_orgs=6,
        num_sybil_clusters=2,
        num_sybil_orgs_2=6,
        num_sybil_bridge_orgs=2,
        config=config,
        rng=rng,
    )
    roles = nx.get_node_attributes(graph, "role")
    assert "sybil_sybil_bridge" in roles.values()
    sybil_nodes = [n for n, role in roles.items() if role == "sybil"]
    assert any(graph.nodes[n].get("sybil_cluster") == 1 for n in sybil_nodes)
    assert any(graph.nodes[n].get("sybil_cluster") == 2 for n in sybil_nodes)


def test_sybil_attack_two_clusters_with_links_smoke():
    rng = random.Random(2)
    config = SybilAttackConfig(
        original_edge_probability=0.7,
        sybil_sybil_edge_probability=0.7,
        sybil2_sybil2_edge_probability=0.7,
        attacker_to_sybil_edge_probability=0.9,
        sybil_to_sybil_bridge_edge_probability=1.0,
        sybil_bridge_to_sybil2_edge_probability=1.0,
        sybil_bridge_to_sybil_bridge_edge_probability=1.0,
        sybil2_to_honest_edge_probability=1.0,
        sybil2_to_attacker_edge_probability=1.0,
        sybil2_to_sybil1_edge_probability=1.0,
        sybil2_to_sybil_bridge_edge_probability=1.0,
        sybil1_to_sybil2_edge_probability=1.0,
        connect_attacker_to_attacker=True,
        connect_attacker_to_honest=True,
        connect_sybil_to_honest=True,
        connect_sybil_to_attacker=True,
        connect_sybil_bridge_to_sybil_bridge=True,
        connect_sybil2_to_honest=True,
        connect_sybil2_to_attacker=True,
        connect_sybil2_to_sybil1=True,
        connect_sybil2_to_sybil_bridge=True,
        connect_sybil1_to_sybil2=True,
        max_attempts=200,
    )
    graph = gen_random_sybil_attack_org_graph(
        num_orgs=12,
        num_sybil_orgs=6,
        num_sybil_clusters=2,
        num_sybil_orgs_2=6,
        num_sybil_bridge_orgs=3,
        config=config,
        rng=rng,
    )
    roles = nx.get_node_attributes(graph, "role")
    assert "sybil_sybil_bridge" in roles.values()
    assert any(role == "sybil" for role in roles.values())


def test_sybil_attack_min_quorum_smoke():
    if not HAS_QBF:
        pytest.skip("QBF support not available")
    rng = random.Random(3)
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
        num_orgs=8,
        num_sybil_orgs=4,
        quorum_selection="min",
        config=config,
        rng=rng,
    )
    roles = nx.get_node_attributes(graph, "role")
    assert "honest" in roles.values()


def test_trustrank_seed_bias():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("seed", "b"),
    ])
    scores = compute_trustrank_scores(
        graph,
        ["seed"],
        alpha=0.2,
        epsilon=1e-10,
        max_iters=1000,
    )
    assert scores["seed"] > scores["a"]
    assert scores["seed"] > scores["b"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_maxflow_scores_simple_chain():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
    ])
    scores = compute_maxflow_scores(graph, ["seed"])
    assert scores["seed"] == 0.0
    assert scores["a"] == 1.0
    assert scores["b"] == 1.0


def test_maxflow_scores_multiple_seeds():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("s1", "a"),
        ("s2", "a"),
    ])
    scores = compute_maxflow_scores(graph, ["s1", "s2"])
    assert scores["s1"] == 0.0
    assert scores["s2"] == 0.0
    assert scores["a"] == 2.0


def test_maxflow_scores_converging_paths():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("s1", "a"),
        ("s2", "a"),
        ("a", "b"),
    ])
    scores = compute_maxflow_scores(graph, ["s1", "s2"])
    assert scores["s1"] == 0.0
    assert scores["s2"] == 0.0
    assert scores["a"] == 2.0
    assert scores["b"] == 1.0


def test_maxflow_scores_sweep_stops():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
    ])
    scores, _capacities, _bcs = compute_maxflow_scores_sweep(
        graph,
        ["seed"],
        seed_capacity=0.25,
        sweep_factor=2.0,
        sweep_bimodality_threshold=1e-6,
        sweep_max_steps=0,
    )
    assert scores["seed"] == 0.0
    assert scores["a"] == 0.25
    assert scores["b"] == 0.25


def test_maxflow_scores_sweep_stops_after_first_iteration(monkeypatch):
    calls: list[float] = []

    def fake_scores(_graph, _seeds, *, seed_capacity, mode):
        calls.append(seed_capacity)
        return {
            "seed": 0.0,
            "a": seed_capacity,
            "b": seed_capacity,
            "c": seed_capacity,
        }

    monkeypatch.setattr(sybil_detection, "compute_maxflow_scores", fake_scores)
    monkeypatch.setattr(
        sybil_detection,
        "compute_bimodality_coefficient",
        lambda _values: 1.0,
    )

    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
        ("a", "c"),
    ])
    scores, capacities, bcs = sybil_detection.compute_maxflow_scores_sweep(
        graph,
        ["seed"],
        seed_capacity=0.25,
        sweep_factor=2.0,
        sweep_bimodality_threshold=0.5,
        sweep_max_steps=5,
    )

    assert calls == [0.25]
    assert capacities == [0.25]
    assert bcs == [1.0]
    assert scores["a"] == 0.25


def test_maxflow_scores_sweep_runs_post_threshold_steps(monkeypatch):
    calls: list[float] = []

    def fake_scores(_graph, _seeds, *, seed_capacity, mode):
        calls.append(seed_capacity)
        return {
            "seed": 0.0,
            "a": seed_capacity,
            "b": seed_capacity,
            "c": seed_capacity,
        }

    monkeypatch.setattr(sybil_detection, "compute_maxflow_scores", fake_scores)
    monkeypatch.setattr(
        sybil_detection,
        "compute_bimodality_coefficient",
        lambda _values: 1.0,
    )

    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
        ("a", "c"),
    ])
    scores, capacities, bcs = sybil_detection.compute_maxflow_scores_sweep(
        graph,
        ["seed"],
        seed_capacity=0.25,
        sweep_factor=2.0,
        sweep_bimodality_threshold=0.5,
        sweep_max_steps=5,
        sweep_post_threshold_steps=2,
    )

    assert calls == [0.25, 0.5, 1.0]
    assert capacities == [0.25, 0.5, 1.0]
    assert bcs == [1.0, 1.0, 1.0]
    assert scores["a"] == 1.0


def test_is_top_tier_complete_digraph():
    graph = nx.complete_graph(6, create_using=nx.DiGraph)
    assert is_top_tier(graph)


def test_is_top_tier_regular_missing_cycle():
    n = 6
    graph = nx.complete_graph(n, create_using=nx.DiGraph)
    for i in range(n):
        graph.remove_edge(i, (i + 1) % n)
    assert is_top_tier(graph)


def test_is_top_tier_outlier_fails():
    n = 6
    graph = nx.complete_graph(n, create_using=nx.DiGraph)
    for i in range(1, n):
        graph.remove_edge(0, i)
        graph.remove_edge(i, 0)
    assert not is_top_tier(graph)


def test_is_top_tier_random_generated_graphs():
    rng = random.Random(123)
    graphs = [
        gen_random_top_tier_org_graph(12, edge_probability=0.85, rng=rng)
        for _ in range(3)
    ]
    assert all(is_top_tier(graph) for graph in graphs)


def test_extract_non_sybil_cluster_maxflow_sweep_high_scores(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    fake_scores = {"a": 0.1, "b": 0.2, "c": 1.5, "d": 1.6}

    def fake_sweep(_graph, _seeds, **_kwargs):
        return fake_scores, [1.0], [0.7]

    monkeypatch.setattr(sybil_detection, "compute_maxflow_scores_sweep", fake_sweep)
    cluster, scores, capacities, bcs = extract_non_sybil_cluster_maxflow_sweep(
        graph,
        ["seed"],
    )

    assert cluster == {"c", "d"}
    assert scores == fake_scores
    assert capacities == [1.0]
    assert bcs == [0.7]


def test_extract_non_sybil_cluster_maxflow_sweep_all_equal(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    fake_scores = {"a": 1.0, "b": 1.0, "c": 1.0}

    def fake_sweep(_graph, _seeds, **_kwargs):
        return fake_scores, [1.0], [0.0]

    monkeypatch.setattr(sybil_detection, "compute_maxflow_scores_sweep", fake_sweep)
    cluster, _scores, _capacities, _bcs = extract_non_sybil_cluster_maxflow_sweep(
        graph,
        ["seed"],
    )

    assert cluster == {"a", "b", "c"}


def test_monte_carlo_seed_reachability_deterministic():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
    ])
    graph.add_node("c")

    scores = estimate_seed_reachability_monte_carlo(
        graph,
        ["seed"],
        failure_prob=0.0,
        trials=5,
        rng=random.Random(0),
    )
    assert scores["seed"] == 1.0
    assert scores["a"] == 1.0
    assert scores["b"] == 1.0
    assert scores["c"] == 0.0


def test_monte_carlo_seed_reachability_all_fail():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
    ])
    scores = estimate_seed_reachability_monte_carlo(
        graph,
        ["seed"],
        failure_prob=1.0,
        trials=5,
        rng=random.Random(0),
    )
    assert scores["seed"] == 1.0
    assert scores["a"] == 0.0
    assert scores["b"] == 0.0


def test_monte_carlo_seed_reachability_remove_seeds():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("seed", "a"),
        ("a", "b"),
    ])
    scores = estimate_seed_reachability_monte_carlo(
        graph,
        ["seed"],
        failure_prob=1.0,
        trials=5,
        remove_seeds=True,
        rng=random.Random(0),
    )
    assert scores["seed"] == 0.0
    assert scores["a"] == 0.0
    assert scores["b"] == 0.0


def test_is_top_tier_small_top_tier_json():
    fbas = load_fbas_from_test_file("top_tier.json")
    org_graph = fbas_to_org_graph(fbas)

    assert org_graph is not None
    assert is_top_tier(org_graph)
