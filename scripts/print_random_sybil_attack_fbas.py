from __future__ import annotations

import argparse
import random

import networkx as nx

from python_fbas.fbas_generator import (
    SybilAttackConfig,
    gen_random_sybil_attack_org_graph,
    top_tier_org_graph_to_fbas_graph,
)
from python_fbas.serialization import serialize


def plot_top_tier_org_graph(graph: nx.DiGraph, *, seed: int | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = {
        node: f"{node}\nT={graph.nodes[node].get('threshold')}"
        for node in graph.nodes
    }
    solid_edges = []
    solid_edge_colors = []
    honest_edges = []
    attacker_sybil_edges = []
    dashed_edges = []
    dotted_edges = []
    for source, target in graph.edges:
        source_role = graph.nodes[source].get("role")
        target_role = graph.nodes[target].get("role")
        if (
            (source_role == "attacker" and target_role == "sybil")
            or (source_role == "honest" and target_role == "attacker")
        ):
            attacker_sybil_edges.append((source, target))
        elif (
            source_role == "sybil"
            and target_role == "honest"
        ):
            dashed_edges.append((source, target))
        elif (
            source_role == "sybil"
            and target_role == "sybil"
        ):
            dotted_edges.append((source, target))
        elif source_role == "honest" and target_role == "honest":
            honest_edges.append((source, target))
        else:
            solid_edges.append((source, target))
            solid_edge_colors.append("#444444")
    honest_nodes = [node for node in graph.nodes
                    if graph.nodes[node].get("role") == "honest"]
    attacker_nodes = [node for node in graph.nodes
                      if graph.nodes[node].get("role") == "attacker"]
    sybil_nodes = [node for node in graph.nodes
                   if graph.nodes[node].get("role") == "sybil"]
    pos = {}
    if honest_nodes:
        honest_graph = graph.subgraph(honest_nodes)
        honest_k = 1.2 if honest_graph.number_of_nodes() > 1 else 0.1
        pos_honest = nx.spring_layout(
            honest_graph,
            seed=seed,
            k=honest_k,
            iterations=200,
        )
        pos.update({node: (coord[0] * 1.2 - 2.2, coord[1] * 1.2)
                    for node, coord in pos_honest.items()})
    if attacker_nodes:
        attacker_graph = graph.subgraph(attacker_nodes)
        attacker_k = 1.2 if attacker_graph.number_of_nodes() > 1 else 0.1
        pos_attackers = nx.spring_layout(
            attacker_graph,
            seed=seed,
            k=attacker_k,
            iterations=200,
        )
        pos.update({node: (coord[0] * 1.2 + 0.3, coord[1] * 1.2)
                    for node, coord in pos_attackers.items()})
    if sybil_nodes:
        sybil_graph = graph.subgraph(sybil_nodes)
        sybil_k = 1.2 if sybil_graph.number_of_nodes() > 1 else 0.1
        pos_sybil = nx.spring_layout(
            sybil_graph,
            seed=seed,
            k=sybil_k,
            iterations=200,
        )
        pos.update({node: (coord[0] * 1.2 + 2.7, coord[1] * 1.2)
                    for node, coord in pos_sybil.items()})
    plt.figure(figsize=(10, 7))
    original_nodes = [node for node in graph.nodes
                      if graph.nodes[node].get("role") != "sybil"]
    sybil_nodes = [node for node in graph.nodes
                   if graph.nodes[node].get("role") == "sybil"]
    if original_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=original_nodes,
            node_size=900,
            node_color="#cfe8ff",
            edgecolors="#111111",
            linewidths=1.2,
        )
    if sybil_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=sybil_nodes,
            node_size=900,
            node_color="#cfe8ff",
        )
    if honest_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=16,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.1",
            edgelist=honest_edges,
            edge_color="#444444",
            width=2.0,
        )
    if attacker_sybil_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=16,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.1",
            edgelist=attacker_sybil_edges,
            edge_color="#c62828",
            width=1.6,
        )
    if solid_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=16,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.1",
            edgelist=solid_edges,
            edge_color=solid_edge_colors,
        )
    if dashed_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=16,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.1",
            edgelist=dashed_edges,
            edge_color="#444444",
            style=(0, (7, 5)),
        )
    if dotted_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=16,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.1",
            edgelist=dotted_edges,
            edge_color="#444444",
            style="dotted",
        )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9)
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#cfe8ff",
            markeredgecolor="#111111",
            markersize=10,
            label="Original org",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#cfe8ff",
            markeredgecolor="#cfe8ff",
            markersize=10,
            label="Sybil org",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.0,
            label="Honest -> honest",
        ),
        Line2D(
            [0],
            [0],
            color="#c62828",
            linewidth=1.6,
            label="Attacker -> sybil / honest -> attacker",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=(0, (7, 5)),
            label="Sybil -> honest",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="dotted",
            label="Sybil -> sybil",
        ),
    ]
    plt.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random FBAS with a Sybil attack topology.")
    parser.add_argument("--orgs", type=int, default=5,
                        help="Number of original orgs")
    parser.add_argument("--sybils", type=int, default=3,
                        help="Number of Sybil orgs")
    parser.add_argument("--edge-probability", type=float, default=0.5,
                        help="Probability of an original-org edge")
    parser.add_argument("--sybil-edge-probability", type=float, default=0.5,
                        help="Probability of a Sybil-org edge")
    parser.add_argument("--attacker-to-sybil-edge-probability",
                        type=float, default=0.5,
                        help="Probability of attacker -> Sybil edges")
    parser.add_argument("--attacker-to-attacker-edge-probability",
                        type=float, default=0.5,
                        help="Probability of attacker -> attacker edges")
    parser.add_argument("--attacker-to-honest-edge-probability",
                        type=float, default=0.5,
                        help="Probability of attacker -> honest edges")
    parser.add_argument("--sybil-to-honest-edge-probability",
                        type=float, default=0.5,
                        help="Probability of Sybil -> honest edges")
    parser.add_argument("--sybil-to-attacker-edge-probability",
                        type=float, default=0.5,
                        help="Probability of Sybil -> attacker edges")
    parser.add_argument("--connect-attacker-to-attacker", action="store_true",
                        help="Connect attackers to each other")
    parser.add_argument("--connect-attacker-to-honest", action="store_true",
                        help="Connect attackers to honest orgs")
    parser.add_argument("--connect-sybil-to-honest", action="store_true",
                        help="Connect Sybil orgs to honest orgs")
    parser.add_argument("--connect-sybil-to-attacker", action="store_true",
                        help="Connect Sybil orgs to attacker orgs")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the top-tier org graph")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (optional)")
    args = parser.parse_args()

    if args.orgs < 2:
        raise ValueError("--orgs must be at least 2")

    if args.sybils < 2:
        raise ValueError("--sybils must be at least 2")

    rng = random.Random(args.seed) if args.seed is not None else None
    config = SybilAttackConfig(
        edge_probability=args.edge_probability,
        sybil_edge_probability=args.sybil_edge_probability,
        attacker_to_sybil_edge_probability=args.attacker_to_sybil_edge_probability,
        attacker_to_attacker_edge_probability=(
            args.attacker_to_attacker_edge_probability
        ),
        attacker_to_honest_edge_probability=args.attacker_to_honest_edge_probability,
        sybil_to_honest_edge_probability=args.sybil_to_honest_edge_probability,
        sybil_to_attacker_edge_probability=args.sybil_to_attacker_edge_probability,
        connect_attacker_to_attacker=args.connect_attacker_to_attacker,
        connect_attacker_to_honest=args.connect_attacker_to_honest,
        connect_sybil_to_honest=args.connect_sybil_to_honest,
        connect_sybil_to_attacker=args.connect_sybil_to_attacker,
    )
    graph = gen_random_sybil_attack_org_graph(
        num_orgs=args.orgs,
        num_sybil_orgs=args.sybils,
        config=config,
        rng=rng,
    )
    fbas = top_tier_org_graph_to_fbas_graph(graph)
    print(serialize(fbas, format="stellarbeat"))
    if args.plot:
        plot_top_tier_org_graph(graph, seed=args.seed)


if __name__ == "__main__":
    main()
