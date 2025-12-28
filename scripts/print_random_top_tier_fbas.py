from __future__ import annotations

import argparse
import random

import networkx as nx

from python_fbas.fbas_generator import (
    gen_random_top_tier_org_graph,
    top_tier_org_graph_to_fbas_graph,
)
from python_fbas.serialization import serialize


def plot_top_tier_org_graph(graph: nx.DiGraph, *, seed: int | None) -> None:
    import matplotlib.pyplot as plt

    labels = {
        node: f"{node}\nT={graph.nodes[node].get('threshold')}"
        for node in graph.nodes
    }
    pos = nx.spring_layout(graph, seed=seed)
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="#cfe8ff")
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        min_source_margin=16,
        min_target_margin=18,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orgs", type=int, default=5)
    parser.add_argument("--edge-probability", type=float, default=0.66)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    top_tier = gen_random_top_tier_org_graph(
        args.orgs,
        edge_probability=args.edge_probability,
        rng=rng,
    )
    fbas = top_tier_org_graph_to_fbas_graph(top_tier)
    print(serialize(fbas, format="stellarbeat"))
    if args.plot:
        plot_top_tier_org_graph(top_tier, seed=args.seed)


if __name__ == "__main__":
    main()
