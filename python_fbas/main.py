"""
Main CLI for the FBAS analysis tool
"""

from collections.abc import Collection
import json
import argparse
import logging
import sys
from typing import Any, Dict, List
from datetime import datetime
import random
import os

import networkx as nx
import yaml
from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import (
    find_disjoint_quorums,
    find_minimal_splitting_set, find_minimal_blocking_set,
    min_history_loss_critical_set, find_min_quorum, top_tier, max_scc,
    random_quorum
)
from python_fbas.pubnet_data import get_pubnet_config
from python_fbas.solver import solvers
from python_fbas.config import (
    update as update_config,
    get as get_config,
    load_config_file,
    load_from_file,
    to_yaml,
)
from python_fbas.serialization import deserialize, serialize
from python_fbas.fbas_generator import (
    SybilAttackConfig,
    gen_random_sybil_attack_org_graph,
    gen_random_top_tier_org_graph,
    top_tier_org_graph_to_fbas_graph,
)
from python_fbas.sybil_detection import (
    compute_maxflow_scores,
    compute_maxflow_scores_sweep,
    compute_trust_scores,
    compute_trustrank_scores,
)

GENERATOR_DEFAULTS: dict[str, Any] = {
    "orgs": 5,
    "sybils": 3,
    "sybils_cluster_2": 3,
    "num_sybil_clusters": 1,
    "sybil_bridge_orgs": 2,
    "quorum_selection": "random",
    "record_run": True,
    "runs_dir": "runs",
    "original_edge_probability": 0.5,
    "sybil_sybil_edge_probability": 0.5,
    "sybil2_sybil2_edge_probability": 0.5,
    "attacker_to_sybil_edge_probability": 0.5,
    "attacker_to_attacker_edge_probability": 0.5,
    "attacker_to_honest_edge_probability": 0.5,
    "sybil_to_honest_edge_probability": 0.5,
    "sybil_to_attacker_edge_probability": 0.5,
    "sybil_to_sybil_bridge_edge_probability": 0.5,
    "sybil_bridge_to_sybil2_edge_probability": 0.5,
    "sybil_bridge_to_sybil_bridge_edge_probability": 0.5,
    "sybil2_to_honest_edge_probability": 0.5,
    "sybil2_to_attacker_edge_probability": 0.5,
    "sybil2_to_sybil1_edge_probability": 0.5,
    "sybil2_to_sybil_bridge_edge_probability": 0.5,
    "sybil1_to_sybil2_edge_probability": 0.5,
    "connect_attacker_to_attacker": False,
    "connect_attacker_to_honest": False,
    "connect_sybil_to_honest": False,
    "connect_sybil_to_attacker": False,
    "connect_sybil_bridge_to_sybil_bridge": False,
    "connect_sybil2_to_honest": False,
    "connect_sybil2_to_attacker": False,
    "connect_sybil2_to_sybil1": False,
    "connect_sybil2_to_sybil_bridge": False,
    "connect_sybil1_to_sybil2": False,
    "seed": None,
}

SYBIL_DETECTION_DEFAULTS: dict[str, Any] = {
    "seed_count": 3,
    "trust_steps": 5,
    "trust_capacity": 1.0,
    "trustrank_alpha": 0.2,
    "trustrank_epsilon": 1e-8,
    "maxflow_seed_capacity": 1.0,
    "maxflow_mode": "standard",
    "maxflow_sweep": False,
    "maxflow_sweep_factor": 2.0,
    "maxflow_sweep_bimodality_threshold": 0.5555555555555556,
    "maxflow_sweep_post_threshold_steps": 0,
    "maxflow_sweep_max_steps": 8,
}


def _generator_defaults_yaml() -> str:
    yaml_lines = [
        "# python-fbas generator configuration file",
        "# Use with: python-fbas random-sybil-attack-fbas --generator-config=FILE",
        "# Add --print-fbas to output the generated FBAS JSON",
        "",
    ]
    for key, value in GENERATOR_DEFAULTS.items():
        value_yaml = yaml.safe_dump(value, default_flow_style=False).strip()
        if value_yaml.endswith("..."):
            value_yaml = value_yaml[:-3].strip()
        yaml_lines.append(f"{key}: {value_yaml}")
    return "\n".join(yaml_lines)


def _sybil_detection_defaults_yaml() -> str:
    yaml_lines = [
        "# python-fbas sybil-detection configuration file",
        "# Use with: python-fbas show-sybil-detection-config",
        "",
    ]
    for key, value in SYBIL_DETECTION_DEFAULTS.items():
        value_yaml = yaml.safe_dump(value, default_flow_style=False).strip()
        if value_yaml.endswith("..."):
            value_yaml = value_yaml[:-3].strip()
        yaml_lines.append(f"{key}: {value_yaml}")
    return "\n".join(yaml_lines)


def _create_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = os.path.join(base_dir, timestamp)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{timestamp}-{counter}")
        counter += 1
    os.makedirs(candidate)
    return candidate


def _write_yaml(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _is_subpath(path: str, parent: str) -> bool:
    path_abs = os.path.abspath(path)
    parent_abs = os.path.abspath(parent)
    return os.path.commonpath([path_abs, parent_abs]) == parent_abs


def _load_json_from_file(validators_file: str) -> List[Dict[str, Any]]:
    with open(validators_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_fbas_graph(args: Any) -> FBASGraph:
    cfg = get_config()

    # Determine data source
    if args.fbas:
        # User specified --fbas
        if args.fbas.startswith('http://') or args.fbas.startswith('https://'):
            data_source = args.fbas
            update_cache = getattr(args, 'update_cache', False)
            logging.info(f"Using Stellar network data from: {data_source}")
            if update_cache:
                logging.info("Forcing cache update...")
            try:
                pubnet_data = get_pubnet_config(update=update_cache, url=args.fbas)
                return deserialize(json.dumps(pubnet_data))
            except (ValueError, IOError) as e:
                logging.error(f"Error: {e}")
                sys.exit(1)
        else:
            # Local file
            logging.info(f"Using local FBAS file: {args.fbas}")
            try:
                json_data = json.dumps(_load_json_from_file(args.fbas))
                return deserialize(json_data)
            except FileNotFoundError:
                logging.error(f"Error: File not found: {args.fbas}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"Error: Could not load file {args.fbas}: {e}")
                sys.exit(1)
    else:
        # Use default URL from config
        data_source = cfg.stellar_data_url
        update_cache = getattr(args, 'update_cache', False)
        logging.info(f"Using default Stellar network data from: {data_source}")
        if update_cache:
            logging.info("Forcing cache update...")
        try:
            pubnet_data = get_pubnet_config(update=update_cache)
            return deserialize(json.dumps(pubnet_data))
        except (ValueError, IOError) as e:
            logging.error(f"Error: {e}")
            sys.exit(1)


def _format_validators(fbas: FBASGraph, vs: Collection[str]) -> list[str]:
    return [fbas.format_validator(v) for v in vs]


def _command_update_cache(args: Any) -> None:
    cfg = get_config()

    # Determine which URL to update cache for
    if args.fbas and (args.fbas.startswith('http://')
                      or args.fbas.startswith('https://')):
        # Update cache for specific URL
        url = args.fbas
        try:
            get_pubnet_config(update=True, url=url)
            logging.info(f"Successfully updated cache for: {url}")
        except (ValueError, IOError) as e:
            logging.error(f"Error: {e}")
            sys.exit(1)
    elif args.fbas:
        # Invalid: trying to update cache for a file
        logging.error("Error: Cannot update cache for local files. Cache updates only work with URLs.")
        sys.exit(1)
    else:
        # Update cache for default URL
        url = cfg.stellar_data_url
        try:
            get_pubnet_config(update=True)
            logging.info(f"Successfully updated cache for default URL: {url}")
        except (ValueError, IOError) as e:
            logging.error(f"Error: {e}")
            sys.exit(1)


def _command_show_config(args: Any) -> None:
    """Display current effective configuration as YAML."""
    yaml_output = to_yaml()
    print(yaml_output)


def _command_show_generator_config(_args: Any) -> None:
    """Display generator defaults as YAML."""
    print(_generator_defaults_yaml())


def _command_show_sybil_detection_config(_args: Any) -> None:
    """Display sybil-detection defaults as YAML."""
    print(_sybil_detection_defaults_yaml())


def _command_check_intersection(args: Any, fbas: FBASGraph) -> None:
    cfg = get_config()
    if cfg.group_by:
        logging.error("Error: --group-by cannot be used with check-intersection")
        sys.exit(1)
    if args.fast:
        fast_result = fbas.fast_intersection_check()
        print(f"Intersection-check result: {fast_result}")
    else:
        result = find_disjoint_quorums(fbas)
        if result:
            print(
                f"Disjoint quorums: {_format_validators(fbas, result.quorum_a)}\n and {_format_validators(fbas, result.quorum_b)}")
        else:
            print("No disjoint quorums found")


def _command_min_splitting_set(_args: Any, fbas: FBASGraph) -> None:
    cfg = get_config()
    result = find_minimal_splitting_set(fbas)
    if not result:
        print("No splitting set found")
        return
    print(f"Minimal splitting-set cardinality is: {len(result.splitting_set)}")
    print(
        f"Example:\n{_format_validators(fbas, result.splitting_set) if not cfg.group_by else result.splitting_set}\nsplits quorums\n{_format_validators(fbas, result.quorum_a)}\nand\n{_format_validators(fbas, result.quorum_b)}")


def _command_min_blocking_set(_args: Any, fbas: FBASGraph) -> None:
    cfg = get_config()
    result = find_minimal_blocking_set(fbas)
    if not result:
        print("No blocking set found")
        return
    print(f"Minimal blocking-set cardinality is: {len(result)}")
    print(
        f"Example:\n{_format_validators(fbas, result) if not cfg.group_by else result}")


def _command_history_loss(_args: Any, fbas: FBASGraph) -> None:
    cfg = get_config()
    if cfg.group_by:
        logging.error("Error: --group-by cannot be used with history-loss")
        sys.exit(1)
    result = min_history_loss_critical_set(fbas)
    print(
        f"Minimal history-loss critical set cardinality is: {len(result.min_critical_set)}")
    print(
        f"Example min critical set:\n{_format_validators(fbas, result.min_critical_set)}")
    print(
        f"Corresponding history-less quorum:\n {_format_validators(fbas, result.quorum)}")


def _command_min_quorum(_args: Any, fbas: FBASGraph) -> None:
    result = find_min_quorum(fbas)
    print(f"Example min quorum:\n{_format_validators(fbas, result)}")


def _command_top_tier(args: Any, fbas: FBASGraph) -> None:
    from_validator = getattr(args, 'from_validator', None)
    result = top_tier(fbas, from_validator=from_validator)
    print(f"Top tier: {_format_validators(fbas, result)}")


def _command_max_scc(_args: Any, fbas: FBASGraph) -> None:
    cfg = get_config()
    if cfg.group_by:
        logging.error("Error: --group-by cannot be used with max-scc")
        sys.exit(1)
    result = max_scc(fbas)
    print(f"Maximal SCC with a quorum: {_format_validators(fbas, result)}")


def _command_random_quorum(args: Any, fbas: FBASGraph) -> None:
    try:
        result = random_quorum(
            fbas,
            seed=args.seed,
            epsilon=args.epsilon,
            delta=args.delta,
            kappa=args.kappa)
    except ImportError as exc:
        logging.error(str(exc))
        sys.exit(1)

    if not result:
        print("No quorum found")
        return
    print(f"Random quorum:\n{_format_validators(fbas, result)}")


def _command_to_json(args: Any, fbas: FBASGraph) -> None:
    """Convert the loaded FBAS to JSON format and print to stdout."""
    if args.format == 'python-fbas':
        print(serialize(fbas, format='python-fbas'))
    elif args.format == 'stellarbeat':
        print(serialize(fbas, format='stellarbeat'))
    else:
        logging.error(f"Error: Unknown format '{args.format}'. Must be 'python-fbas' or 'stellarbeat'")
        sys.exit(1)


def _command_validator_metadata(args: Any, fbas: FBASGraph) -> None:
    from python_fbas.stellarbeat_serializer import qset_of

    validator_id = args.validator
    if validator_id not in fbas.get_validators():
        logging.error(f"Error: Unknown validator: {validator_id}")
        sys.exit(1)

    attrs = fbas.vertice_attrs(validator_id).copy()
    if 'publicKey' not in attrs:
        attrs['publicKey'] = validator_id
    qset = qset_of(fbas, validator_id)
    if qset is not None:
        attrs['quorumSet'] = qset.to_json()
    print(json.dumps(attrs, indent=2, sort_keys=True))


def _plot_random_org_graph(
    graph: nx.DiGraph,
    *,
    seed: int | None,
    trust_scores: dict[str, float] | None = None,
    trust_seeds: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    layout_seed = None if seed is None else seed % (2**32)
    has_roles = any("role" in graph.nodes[node] for node in graph.nodes)
    if not has_roles:
        graph = graph.copy()
        nx.set_node_attributes(graph, "honest", "role")

    labels = {}
    for node in graph.nodes:
        if trust_scores is not None:
            score = trust_scores.get(node, 0.0)
            labels[node] = f"{node}\n{score:.3f}"
        else:
            labels[node] = f"{node}"
    solid_edges = []
    solid_edge_colors = []
    original_edges = []
    attacker_sybil_edges = []
    dashed_edges = []
    dotted_edges = []
    for source, target in graph.edges:
        source_role = graph.nodes[source].get("role")
        target_role = graph.nodes[target].get("role")
        sybil_roles = {"sybil", "sybil_sybil_bridge"}
        if source_role == "attacker" and target_role in sybil_roles:
            attacker_sybil_edges.append((source, target))
        elif source_role in sybil_roles and target_role == "honest":
            dashed_edges.append((source, target))
        elif source_role in sybil_roles and target_role in sybil_roles:
            dotted_edges.append((source, target))
        elif source_role == "honest" and target_role in {"honest", "attacker"}:
            original_edges.append((source, target))
        else:
            solid_edges.append((source, target))
            solid_edge_colors.append("#444444")
    honest_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") == "honest"
    ]
    attacker_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") == "attacker"
    ]
    sybil_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") == "sybil"
    ]
    bridge_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") == "sybil_sybil_bridge"
    ]
    sybil1_nodes = [
        node for node in sybil_nodes
        if graph.nodes[node].get("sybil_cluster", 1) == 1
    ]
    sybil2_nodes = [
        node for node in sybil_nodes
        if graph.nodes[node].get("sybil_cluster") == 2
    ]
    pos = {}
    if honest_nodes and not attacker_nodes and not sybil_nodes and not bridge_nodes:
        pos = nx.spring_layout(graph, seed=layout_seed)
    else:
        def _layout_group(nodes: list[str], x_offset: float) -> None:
            if not nodes:
                return
            subgraph = graph.subgraph(nodes)
            group_k = 1.2 if subgraph.number_of_nodes() > 1 else 0.1
            group_pos = nx.spring_layout(
                subgraph,
                seed=layout_seed,
                k=group_k,
                iterations=200,
            )
            pos.update({
                node: (coord[0] * 1.2 + x_offset, coord[1] * 1.2)
                for node, coord in group_pos.items()
            })

        _layout_group(honest_nodes, -3.2)
        _layout_group(attacker_nodes, -1.0)
        _layout_group(sybil1_nodes, 1.2)
        _layout_group(bridge_nodes, 3.4)
        _layout_group(sybil2_nodes, 5.6)
    if not pos:
        pos = nx.spring_layout(graph, seed=layout_seed)
    plt.figure(figsize=(10, 7))
    original_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") in {"honest", "attacker"}
    ]
    sybil_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") == "sybil"
    ]
    bridge_nodes = [
        node for node in graph.nodes
        if graph.nodes[node].get("role") == "sybil_sybil_bridge"
    ]
    if original_nodes:
        node_colors = None
        linewidths = 1.2
        edgecolors = []
        if trust_scores:
            max_score = max(trust_scores.values(), default=0.0)
            node_colors = []
            linewidths = []
        for node in original_nodes:
            if trust_scores:
                score = trust_scores.get(node, 0.0)
                if max_score > 0:
                    intensity = 0.3 + 0.7 * (score / max_score)
                else:
                    intensity = 0.3
                node_colors.append(plt.cm.Blues(intensity))
                linewidths.append(1.2)
            role = graph.nodes[node].get("role")
            edgecolors.append("#c62828" if role == "attacker" else "#111111")
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=original_nodes,
            node_size=900,
            node_color=node_colors if node_colors else "#cfe8ff",
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
    if sybil_nodes:
        node_colors = None
        linewidths = 0.0
        edgecolors = "none"
        if trust_scores:
            max_score = max(trust_scores.values(), default=0.0)
            node_colors = []
            linewidths = []
            for node in sybil_nodes:
                score = trust_scores.get(node, 0.0)
                if max_score > 0:
                    intensity = 0.3 + 0.7 * (score / max_score)
                else:
                    intensity = 0.3
                node_colors.append(plt.cm.Blues(intensity))
                linewidths.append(0.0)
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=sybil_nodes,
            node_size=900,
            node_color=node_colors if node_colors else "#cfe8ff",
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
    if bridge_nodes:
        node_colors = None
        linewidths = 1.2
        edgecolors = "#8d6e63"
        if trust_scores:
            max_score = max(trust_scores.values(), default=0.0)
            node_colors = []
            linewidths = []
            for node in bridge_nodes:
                score = trust_scores.get(node, 0.0)
                if max_score > 0:
                    intensity = 0.3 + 0.7 * (score / max_score)
                else:
                    intensity = 0.3
                node_colors.append(plt.cm.Blues(intensity))
                linewidths.append(1.2)
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=bridge_nodes,
            node_size=900,
            node_color=node_colors if node_colors else "#f4e0d6",
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
    if original_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=16,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.1",
            edgelist=original_edges,
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
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=9,
        font_color="#f7f7f7",
    )
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#cfe8ff",
            markeredgecolor="#111111",
            markersize=10,
            label="Honest org",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#cfe8ff",
            markeredgecolor="#c62828",
            markersize=10,
            label="Attacker org",
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
            marker="o",
            color="none",
            markerfacecolor="#f4e0d6",
            markeredgecolor="#8d6e63",
            markersize=10,
            label="Sybil-bridge org",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.0,
            label="Original edge",
        ),
        Line2D(
            [0],
            [0],
            color="#c62828",
            linewidth=1.6,
            label="Attacker -> sybil",
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
    if trust_seeds:
        trust_seed_label = ", ".join(trust_seeds)
        legend_items.append(
            Line2D(
                [],
                [],
                color="none",
                label=f"Trust seeds: {trust_seed_label}",
            )
        )
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


def _command_random_sybil_attack_fbas(args: Any) -> None:
    default_params = GENERATOR_DEFAULTS
    allowed_keys = set(GENERATOR_DEFAULTS.keys())
    config_params = {}
    generator_config_path = args.generator_config
    if generator_config_path is None:
        config_dirs = []
        if args.config_dir:
            config_dirs.append(args.config_dir)
        config_dirs.append(".")
        for config_dir in config_dirs:
            default_generator_config = os.path.join(
                config_dir,
                "python-fbas.generator.cfg",
            )
            if os.path.exists(default_generator_config):
                generator_config_path = default_generator_config
                break
    if generator_config_path:
        config_params = load_from_file(generator_config_path)
        if not isinstance(config_params, dict):
            raise ValueError(
                "--generator-config must contain a YAML mapping of parameters")
        invalid_keys = set(config_params.keys()) - allowed_keys
        if invalid_keys:
            logging.warning(
                "Ignoring unknown generator config keys: %s",
                sorted(invalid_keys),
            )
            config_params = {
                key: value
                for key, value in config_params.items()
                if key in allowed_keys
            }

    params = default_params.copy()
    params.update(config_params)
    if args.original_edge_probability is not None:
        params["original_edge_probability"] = args.original_edge_probability
    if args.sybil_sybil_edge_probability is not None:
        params["sybil_sybil_edge_probability"] = args.sybil_sybil_edge_probability
    for key in allowed_keys:
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value

    if params["orgs"] < 2:
        raise ValueError("--orgs must be at least 2")

    num_sybil_clusters = params["num_sybil_clusters"]
    if num_sybil_clusters not in {0, 1, 2}:
        raise ValueError("--num-sybil-clusters must be 0, 1, or 2")
    if num_sybil_clusters >= 1 and params["sybils"] < 2:
        raise ValueError("--sybils must be at least 2 when Sybil clusters are enabled")
    if num_sybil_clusters == 2:
        if params["sybils_cluster_2"] < 2:
            raise ValueError(
                "--sybils-cluster-2 must be at least 2 when using two Sybil clusters"
            )
        if params["sybil_bridge_orgs"] < 1:
            raise ValueError(
                "--sybil-bridge-orgs must be at least 1 when using two Sybil clusters"
            )
    if params["quorum_selection"] not in {"random", "min"}:
        raise ValueError("--quorum-selection must be 'random' or 'min'")
    if params["record_run"] and not params["runs_dir"]:
        raise ValueError("--runs-dir must be set when --record-run is enabled")

    seed = params["seed"]
    if seed is None:
        seed = random.SystemRandom().randint(1, 2**63 - 1)
        print(f"Random seed: {seed}", file=sys.stderr)
        params["seed"] = seed
    rng = random.Random(seed)
    if num_sybil_clusters == 0:
        graph = gen_random_top_tier_org_graph(
            params["orgs"],
            edge_probability=params["original_edge_probability"],
            rng=rng,
        )
    else:
        config = SybilAttackConfig(
            original_edge_probability=params["original_edge_probability"],
            sybil_sybil_edge_probability=params["sybil_sybil_edge_probability"],
            sybil2_sybil2_edge_probability=params["sybil2_sybil2_edge_probability"],
            attacker_to_sybil_edge_probability=(
                params["attacker_to_sybil_edge_probability"]
            ),
            attacker_to_attacker_edge_probability=(
                params["attacker_to_attacker_edge_probability"]
            ),
            attacker_to_honest_edge_probability=(
                params["attacker_to_honest_edge_probability"]
            ),
            sybil_to_honest_edge_probability=(
                params["sybil_to_honest_edge_probability"]
            ),
            sybil_to_attacker_edge_probability=(
                params["sybil_to_attacker_edge_probability"]
            ),
            sybil_to_sybil_bridge_edge_probability=(
                params["sybil_to_sybil_bridge_edge_probability"]
            ),
            sybil_bridge_to_sybil2_edge_probability=(
                params["sybil_bridge_to_sybil2_edge_probability"]
            ),
            sybil_bridge_to_sybil_bridge_edge_probability=(
                params["sybil_bridge_to_sybil_bridge_edge_probability"]
            ),
            sybil2_to_honest_edge_probability=(
                params["sybil2_to_honest_edge_probability"]
            ),
            sybil2_to_attacker_edge_probability=(
                params["sybil2_to_attacker_edge_probability"]
            ),
            sybil2_to_sybil1_edge_probability=(
                params["sybil2_to_sybil1_edge_probability"]
            ),
            sybil2_to_sybil_bridge_edge_probability=(
                params["sybil2_to_sybil_bridge_edge_probability"]
            ),
            sybil1_to_sybil2_edge_probability=(
                params["sybil1_to_sybil2_edge_probability"]
            ),
            connect_attacker_to_attacker=params["connect_attacker_to_attacker"],
            connect_attacker_to_honest=params["connect_attacker_to_honest"],
            connect_sybil_to_honest=params["connect_sybil_to_honest"],
            connect_sybil_to_attacker=params["connect_sybil_to_attacker"],
            connect_sybil_bridge_to_sybil_bridge=(
                params["connect_sybil_bridge_to_sybil_bridge"]
            ),
            connect_sybil2_to_honest=params["connect_sybil2_to_honest"],
            connect_sybil2_to_attacker=params["connect_sybil2_to_attacker"],
            connect_sybil2_to_sybil1=params["connect_sybil2_to_sybil1"],
            connect_sybil2_to_sybil_bridge=(
                params["connect_sybil2_to_sybil_bridge"]
            ),
            connect_sybil1_to_sybil2=params["connect_sybil1_to_sybil2"],
        )
        graph = gen_random_sybil_attack_org_graph(
            num_orgs=params["orgs"],
            num_sybil_orgs=params["sybils"],
            num_sybil_clusters=num_sybil_clusters,
            num_sybil_orgs_2=params["sybils_cluster_2"],
            num_sybil_bridge_orgs=params["sybil_bridge_orgs"],
            quorum_selection=params["quorum_selection"],
            config=config,
            rng=rng,
        )
    fbas = top_tier_org_graph_to_fbas_graph(graph)
    if args.print_fbas:
        print(serialize(fbas, format="stellarbeat"))
    sybil_defaults = SYBIL_DETECTION_DEFAULTS
    sybil_allowed = set(SYBIL_DETECTION_DEFAULTS.keys())
    deprecated_sybil_keys = {
        "steps": "trust_steps",
        "capacity": "trust_capacity",
    }
    sybil_params = sybil_defaults.copy()
    sybil_detection_path = args.sybil_detection_config
    if sybil_detection_path is None:
        config_dirs = []
        if args.config_dir:
            config_dirs.append(args.config_dir)
        config_dirs.append(".")
        for config_dir in config_dirs:
            default_sybil_detection = os.path.join(
                config_dir,
                "python-fbas.sybil-detection.cfg",
            )
            if os.path.exists(default_sybil_detection):
                sybil_detection_path = default_sybil_detection
                break
    if sybil_detection_path:
        sybil_config = load_from_file(sybil_detection_path)
        if not isinstance(sybil_config, dict):
            raise ValueError(
                "--sybil-detection-config must contain a YAML mapping of parameters")
        for old_key, new_key in deprecated_sybil_keys.items():
            if old_key in sybil_config:
                if new_key in sybil_config:
                    logging.warning(
                        "Ignoring deprecated sybil-detection key %s because %s is set",
                        old_key,
                        new_key,
                    )
                else:
                    logging.warning(
                        "sybil-detection key %s is deprecated; use %s",
                        old_key,
                        new_key,
                    )
                    sybil_config[new_key] = sybil_config[old_key]
                sybil_config.pop(old_key, None)
        invalid_keys = set(sybil_config.keys()) - sybil_allowed
        if invalid_keys:
            logging.warning(
                "Ignoring unknown sybil-detection config keys: %s",
                sorted(invalid_keys),
            )
            sybil_config = {
                key: value
                for key, value in sybil_config.items()
                if key in sybil_allowed
            }
        sybil_params.update(sybil_config)
    if args.sybil_detection_trust_steps is not None:
        sybil_params["trust_steps"] = args.sybil_detection_trust_steps
    if args.sybil_detection_trust_capacity is not None:
        sybil_params["trust_capacity"] = args.sybil_detection_trust_capacity
    if args.sybil_detection_seed_count is not None:
        sybil_params["seed_count"] = args.sybil_detection_seed_count
    if args.sybil_detection_trustrank_alpha is not None:
        sybil_params["trustrank_alpha"] = args.sybil_detection_trustrank_alpha
    if args.sybil_detection_trustrank_epsilon is not None:
        sybil_params["trustrank_epsilon"] = args.sybil_detection_trustrank_epsilon
    if args.sybil_detection_maxflow_seed_capacity is not None:
        sybil_params["maxflow_seed_capacity"] = (
            args.sybil_detection_maxflow_seed_capacity
        )
    if args.sybil_detection_maxflow_mode is not None:
        sybil_params["maxflow_mode"] = args.sybil_detection_maxflow_mode
    if args.sybil_detection_maxflow_sweep is not None:
        sybil_params["maxflow_sweep"] = args.sybil_detection_maxflow_sweep
    if args.sybil_detection_maxflow_sweep_factor is not None:
        sybil_params["maxflow_sweep_factor"] = (
            args.sybil_detection_maxflow_sweep_factor
        )
    if args.sybil_detection_maxflow_sweep_bimodality_threshold is not None:
        sybil_params["maxflow_sweep_bimodality_threshold"] = (
            args.sybil_detection_maxflow_sweep_bimodality_threshold
        )
    if args.sybil_detection_maxflow_sweep_post_threshold_steps is not None:
        sybil_params["maxflow_sweep_post_threshold_steps"] = (
            args.sybil_detection_maxflow_sweep_post_threshold_steps
        )
    if args.sybil_detection_maxflow_sweep_max_steps is not None:
        sybil_params["maxflow_sweep_max_steps"] = (
            args.sybil_detection_maxflow_sweep_max_steps
        )

    record_run = params["record_run"]
    if record_run and args.config_dir:
        if _is_subpath(args.config_dir, params["runs_dir"]):
            record_run = False
            logging.info(
                "Config dir is inside runs dir; skipping run recording.")

    if record_run:
        run_dir = _create_run_dir(params["runs_dir"])
        _write_yaml(os.path.join(run_dir, "python-fbas.generator.cfg"), params)
        _write_yaml(
            os.path.join(run_dir, "python-fbas.sybil-detection.cfg"),
            sybil_params,
        )
        with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as handle:
            handle.write(" ".join(sys.argv))
            handle.write("\n")

    if sum(
        1
        for enabled in (
            args.plot_with_trust,
            args.plot_with_trustrank,
            args.plot_with_maxflow,
        )
        if enabled
    ) > 1:
        raise ValueError(
            "Choose only one of --plot-with-trust, --plot-with-trustrank, or "
            "--plot-with-maxflow")

    def _choose_trust_seeds() -> list[str]:
        roles = nx.get_node_attributes(graph, "role")
        honest_nodes = [
            node for node in graph.nodes
            if roles.get(node, "honest") == "honest"
        ]
        if not honest_nodes:
            honest_nodes = list(graph.nodes)
        trust_rng = rng if rng is not None else random.Random()
        seed_count = sybil_params["seed_count"]
        if seed_count <= 0:
            raise ValueError("sybil-detection seed_count must be positive")
        if seed_count >= len(honest_nodes):
            return list(honest_nodes)
        return trust_rng.sample(honest_nodes, seed_count)

    if args.plot_with_trust:
        trust_seeds = _choose_trust_seeds()
        trust_scores = compute_trust_scores(
            graph,
            trust_seeds,
            steps=sybil_params["trust_steps"],
            capacity=sybil_params["trust_capacity"],
        )
        _plot_random_org_graph(
            graph,
            seed=params["seed"],
            trust_scores=trust_scores,
            trust_seeds=trust_seeds,
        )
    elif args.plot_with_trustrank:
        trust_seeds = _choose_trust_seeds()
        trust_scores = compute_trustrank_scores(
            graph,
            trust_seeds,
            alpha=sybil_params["trustrank_alpha"],
            epsilon=sybil_params["trustrank_epsilon"],
        )
        _plot_random_org_graph(
            graph,
            seed=params["seed"],
            trust_scores=trust_scores,
            trust_seeds=trust_seeds,
        )
    elif args.plot_with_maxflow:
        trust_seeds = _choose_trust_seeds()
        if sybil_params["maxflow_sweep"]:
            trust_scores, capacities, bcs = compute_maxflow_scores_sweep(
                graph,
                trust_seeds,
                seed_capacity=sybil_params["maxflow_seed_capacity"],
                mode=sybil_params["maxflow_mode"],
                sweep_factor=sybil_params["maxflow_sweep_factor"],
                sweep_bimodality_threshold=(
                    sybil_params["maxflow_sweep_bimodality_threshold"]
                ),
                sweep_post_threshold_steps=(
                    sybil_params["maxflow_sweep_post_threshold_steps"]
                ),
                sweep_max_steps=sybil_params["maxflow_sweep_max_steps"],
            )
            print("Max-flow sweep bimodality coefficients:")
            for capacity, bc in zip(capacities, bcs):
                print(f"  seed_capacity={capacity:.6g} bc={bc:.6f}")
        else:
            trust_scores = compute_maxflow_scores(
                graph,
                trust_seeds,
                seed_capacity=sybil_params["maxflow_seed_capacity"],
                mode=sybil_params["maxflow_mode"],
            )
        _plot_random_org_graph(
            graph,
            seed=params["seed"],
            trust_scores=trust_scores,
            trust_seeds=trust_seeds,
        )
        if sybil_params["maxflow_sweep"]:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.plot(capacities, bcs, marker="o", color="#1f77b4")
            plt.xlabel("Seed capacity")
            plt.ylabel("Bimodality coefficient")
            plt.title("Max-flow sweep bimodality")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    elif args.plot:
        _plot_random_org_graph(graph, seed=params["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FBAS analysis CLI. Use '<command> -h' for subcommand help.",
    )
    # specify log level with --log-level, with default WARNING:
    parser.add_argument(
        '--log-level',
        default='WARNING',
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # specify a data source:
    parser.add_argument(
        '--fbas',
        help="Where to find the description of the FBAS to analyze. Must be a URL (http/https) or a path to a JSON file.")
    parser.add_argument(
        '--reachable-from',
        default=None,
        help="Restrict the FBAS to what's reachable from the provided validator")
    parser.add_argument(
        '--group-by',
        default=None,
        help="Group by the provided field (e.g. min-splitting-set with --group-by=homeDomain will compute the minimum number of home domains to corrupt to create disjoint quorums)")

    parser.add_argument(
        '--validator-display',
        default='both',
        help="How to display validators in output. Can be 'id', 'name', or 'both'")
    parser.add_argument(
        '--update-cache',
        action='store_true',
        help="Force cache update when using a URL")
    parser.add_argument(
        '--config-file',
        default=None,
        help="Path to YAML configuration file. If not specified, python-fbas.cfg in current directory will be used if it exists.")
    parser.add_argument(
        '--config-dir',
        default=None,
        help="Directory containing python-fbas config files")

    parser.add_argument(
        '--cardinality-encoding',
        default='totalizer',
        help="Cardinality encoding, either 'naive' or 'totalizer'")
    parser.add_argument(
        '--sat-solver',
        default='cryptominisat5',
        help=f"SAT solver to use ({solvers}). See the documentation of the pysat package for more information.")
    parser.add_argument(
        '--max-sat-algo',
        default='LSU',
        help="MaxSAT algorithm to use (LSU or RC2)")
    parser.add_argument(
        '--output-problem',
        default=None,
        help="Write the constraint-satisfaction problem to the provided path")

    # subcommands:
    subparsers = parser.add_subparsers(
        dest="command",
        help="sub-command help",
        required=True)

    # Command for updating cached data
    parser_update_cache = subparsers.add_parser(
        'update-cache',
        help="Update cached data. Use --fbas=URL to update cache for a specific URL, or no --fbas to update the default URL cache.")
    parser_update_cache.set_defaults(func=_command_update_cache)

    # Command for showing current configuration
    parser_show_config = subparsers.add_parser(
        'show-config',
        help="Display current effective configuration as YAML")
    parser_show_config.set_defaults(func=_command_show_config)

    parser_show_generator_config = subparsers.add_parser(
        'show-generator-config',
        help="Display generator defaults as YAML")
    parser_show_generator_config.set_defaults(func=_command_show_generator_config)

    parser_show_sybil_detection_config = subparsers.add_parser(
        'show-sybil-detection-config',
        help="Display sybil-detection defaults as YAML")
    parser_show_sybil_detection_config.set_defaults(
        func=_command_show_sybil_detection_config)

    # Command for checking intersection
    parser_is_intertwined = subparsers.add_parser(
        'check-intersection',
        help="Check that the FBAS is intertwined (i.e. whether all quorums intersect)")
    parser_is_intertwined.add_argument(
        '--fast',
        action='store_true',
        help="Use the fast heuristic (which does not use a SAT solver and only returns true, meaning all quorums intersect, or unknown)")
    parser_is_intertwined.set_defaults(func=_command_check_intersection)

    # Command for minimum splitting set
    parser_min_splitting_set = subparsers.add_parser(
        'min-splitting-set', help="Find minimal-cardinality splitting set")
    parser_min_splitting_set.set_defaults(func=_command_min_splitting_set)

    parser_min_blocking_set = subparsers.add_parser(
        'min-blocking-set', help="Find minimal-cardinality blocking set")
    parser_min_blocking_set.set_defaults(func=_command_min_blocking_set)

    parser_history_loss = subparsers.add_parser(
        'history-loss',
        help="Find a minimal-cardinality set of validators such that, should they stop publishing valid history, would allow a full quorum to get ahead without publishing valid history (in which case history may be lost)")
    parser_history_loss.set_defaults(func=_command_history_loss)

    parser_min_quorum = subparsers.add_parser(
        'min-quorum', help="Find minimal-cardinality quorum")
    parser_min_quorum.set_defaults(func=_command_min_quorum)

    parser_top_tier = subparsers.add_parser(
        'top-tier', help="Find the top tier of the FBAS")
    parser_top_tier.add_argument(
        '--from-validator',
        default=None,
        help="Restrict the FBAS to what's reachable from the provided validator before computing top tier")
    parser_top_tier.set_defaults(func=_command_top_tier)

    parser_max_scc = subparsers.add_parser(
        'max-scc', help="Find a maximal strongly-connected component of the FBAS that contains a quorum")
    parser_max_scc.set_defaults(func=_command_max_scc)

    parser_random_quorum = subparsers.add_parser(
        'random-quorum',
        help="Sample a random quorum using UniGen")
    parser_random_quorum.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Seed for UniGen (default: random)")
    parser_random_quorum.add_argument(
        '--epsilon',
        type=float,
        default=0.8,
        help="UniGen tolerance factor (epsilon)")
    parser_random_quorum.add_argument(
        '--delta',
        type=float,
        default=0.2,
        help="UniGen confidence parameter (delta)")
    parser_random_quorum.add_argument(
        '--kappa',
        type=float,
        default=0.638,
        help="UniGen uniformity parameter (kappa)")
    parser_random_quorum.set_defaults(func=_command_random_quorum)

    parser_to_json = subparsers.add_parser(
        'to-json', help="Convert the loaded FBAS to JSON format and print to stdout")
    parser_to_json.add_argument(
        '--format',
        default='python-fbas',
        choices=['python-fbas', 'stellarbeat'],
        help="Output format: 'python-fbas' (default) or 'stellarbeat'")
    parser_to_json.set_defaults(func=_command_to_json)

    parser_validator_metadata = subparsers.add_parser(
        'validator-metadata',
        help="Return the metadata for a validator")
    parser_validator_metadata.add_argument(
        'validator',
        help="Validator public key")
    parser_validator_metadata.set_defaults(func=_command_validator_metadata)

    parser_random_sybil_attack = subparsers.add_parser(
        'random-sybil-attack-fbas',
        help="Generate a random FBAS with a Sybil attack topology")
    parser_random_sybil_attack.add_argument(
        "--generator-config",
        default=None,
        help="Path to YAML config for generator parameters")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-config",
        default=None,
        help="Path to YAML config for sybil-detection parameters")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-trust-steps",
        "--sybil-detection-steps",
        dest="sybil_detection_trust_steps",
        type=int,
        default=None,
        help="Trust-propagation steps for --plot-with-trust")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-trust-capacity",
        "--sybil-detection-capacity",
        dest="sybil_detection_trust_capacity",
        type=float,
        default=None,
        help="Trust-propagation capacity for --plot-with-trust")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-seed-count", type=int, default=None,
        help="Number of trusted seeds for --plot-with-trust")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-trustrank-alpha", type=float, default=None,
        help="TrustRank restart probability for --plot-with-trustrank")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-trustrank-epsilon", type=float, default=None,
        help="TrustRank convergence tolerance for --plot-with-trustrank")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-seed-capacity", type=float, default=None,
        help="Seed node capacity for --plot-with-maxflow")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-mode",
        choices=["standard"],
        default=None,
        help="Max-flow scoring mode for --plot-with-maxflow")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sweep max-flow seed capacity until scores stabilize")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-sweep-factor", type=float, default=None,
        help="Capacity multiplier per sweep step")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-sweep-bimodality-threshold",
        type=float,
        default=None,
        help="Bimodality coefficient threshold to stop the sweep")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-sweep-post-threshold-steps",
        type=int,
        default=None,
        help="Extra sweep steps after hitting the bimodality threshold")
    parser_random_sybil_attack.add_argument(
        "--sybil-detection-maxflow-sweep-max-steps", type=int, default=None,
        help="Maximum number of sweep steps")
    parser_random_sybil_attack.add_argument(
        "--orgs", type=int, default=None,
        help="Number of original orgs")
    parser_random_sybil_attack.add_argument(
        "--sybils", type=int, default=None,
        help="Number of Sybil orgs in cluster 1")
    parser_random_sybil_attack.add_argument(
        "--sybils-cluster-2", type=int, default=None,
        help="Number of Sybil orgs in cluster 2")
    parser_random_sybil_attack.add_argument(
        "--num-sybil-clusters", type=int, default=None,
        help="Number of Sybil clusters (0, 1, or 2)")
    parser_random_sybil_attack.add_argument(
        "--sybil-bridge-orgs", type=int, default=None,
        help="Number of Sybil-bridge orgs between Sybil clusters")
    parser_random_sybil_attack.add_argument(
        "--quorum-selection",
        choices=["random", "min"],
        default=None,
        help="Quorum selection strategy: random (UniGen) or min (QBF)")
    parser_random_sybil_attack.add_argument(
        "--record-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record run configs for reproducibility")
    parser_random_sybil_attack.add_argument(
        "--runs-dir",
        default=None,
        help="Directory to store reproducible run configs")
    parser_random_sybil_attack.add_argument(
        "--original-edge-probability", type=float, default=None,
        help="Probability of an original-org edge")
    parser_random_sybil_attack.add_argument(
        "--sybil-sybil-edge-probability", type=float, default=None,
        help="Probability of a Sybil-org edge")
    parser_random_sybil_attack.add_argument(
        "--sybil2-sybil2-edge-probability", type=float, default=None,
        help="Probability of a Sybil2-org edge")
    parser_random_sybil_attack.add_argument(
        "--attacker-to-sybil-edge-probability",
        type=float, default=None,
        help="Probability of attacker -> Sybil edges")
    parser_random_sybil_attack.add_argument(
        "--attacker-to-attacker-edge-probability",
        type=float, default=None,
        help="Probability of attacker -> attacker edges")
    parser_random_sybil_attack.add_argument(
        "--attacker-to-honest-edge-probability",
        type=float, default=None,
        help="Probability of attacker -> honest edges")
    parser_random_sybil_attack.add_argument(
        "--sybil-to-honest-edge-probability",
        type=float, default=None,
        help="Probability of Sybil -> honest edges")
    parser_random_sybil_attack.add_argument(
        "--sybil-to-attacker-edge-probability",
        type=float, default=None,
        help="Probability of Sybil -> attacker edges")
    parser_random_sybil_attack.add_argument(
        "--sybil-to-sybil-bridge-edge-probability",
        type=float, default=None,
        help="Probability of Sybil -> Sybil-bridge edges")
    parser_random_sybil_attack.add_argument(
        "--sybil-bridge-to-sybil2-edge-probability",
        type=float, default=None,
        help="Probability of Sybil-bridge -> Sybil2 edges")
    parser_random_sybil_attack.add_argument(
        "--sybil-bridge-to-sybil-bridge-edge-probability",
        type=float, default=None,
        help="Probability of Sybil-bridge -> Sybil-bridge edges")
    parser_random_sybil_attack.add_argument(
        "--sybil2-to-honest-edge-probability",
        type=float, default=None,
        help="Probability of Sybil2 -> honest edges")
    parser_random_sybil_attack.add_argument(
        "--sybil2-to-attacker-edge-probability",
        type=float, default=None,
        help="Probability of Sybil2 -> attacker edges")
    parser_random_sybil_attack.add_argument(
        "--sybil2-to-sybil1-edge-probability",
        type=float, default=None,
        help="Probability of Sybil2 -> Sybil1 edges")
    parser_random_sybil_attack.add_argument(
        "--sybil2-to-sybil-bridge-edge-probability",
        type=float, default=None,
        help="Probability of Sybil2 -> Sybil-bridge edges")
    parser_random_sybil_attack.add_argument(
        "--sybil1-to-sybil2-edge-probability",
        type=float, default=None,
        help="Probability of Sybil1 -> Sybil2 edges")
    parser_random_sybil_attack.add_argument(
        "--connect-attacker-to-attacker", action="store_true", default=None,
        help="Connect attackers to each other")
    parser_random_sybil_attack.add_argument(
        "--connect-attacker-to-honest", action="store_true", default=None,
        help="Connect attackers to honest orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil-to-honest", action="store_true", default=None,
        help="Connect Sybil orgs to honest orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil-to-attacker", action="store_true", default=None,
        help="Connect Sybil orgs to attacker orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil-bridge-to-sybil-bridge",
        action="store_true",
        default=None,
        help="Connect Sybil-bridge orgs to each other")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil2-to-honest", action="store_true", default=None,
        help="Connect Sybil2 orgs to honest orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil2-to-attacker", action="store_true", default=None,
        help="Connect Sybil2 orgs to attacker orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil2-to-sybil1", action="store_true", default=None,
        help="Connect Sybil2 orgs to Sybil1 orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil2-to-sybil-bridge", action="store_true", default=None,
        help="Connect Sybil2 orgs to Sybil-bridge orgs")
    parser_random_sybil_attack.add_argument(
        "--connect-sybil1-to-sybil2", action="store_true", default=None,
        help="Connect Sybil1 orgs to Sybil2 orgs")
    parser_random_sybil_attack.add_argument(
        "--plot", action="store_true",
        help="Plot the org graph")
    parser_random_sybil_attack.add_argument(
        "--print-fbas", action="store_true",
        help="Print the generated FBAS JSON to stdout")
    parser_random_sybil_attack.add_argument(
        "--plot-with-trust", action="store_true",
        help="Plot the org graph with trust scores from a random honest org")
    parser_random_sybil_attack.add_argument(
        "--plot-with-trustrank", action="store_true",
        help="Plot the org graph with TrustRank scores from random honest orgs")
    parser_random_sybil_attack.add_argument(
        "--plot-with-maxflow", action="store_true",
        help="Plot the org graph with max-flow scores from random honest orgs")
    parser_random_sybil_attack.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (optional)")
    parser_random_sybil_attack.set_defaults(func=_command_random_sybil_attack_fbas)

    args = parser.parse_args()

    # Set log level early
    debug_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if args.log_level not in debug_levels:
        print(f"Error: Log level must be one of {debug_levels}", file=sys.stderr)
        sys.exit(1)
    logging.getLogger().setLevel(args.log_level)

    if args.config_dir and args.config_file is None:
        config_candidate = os.path.join(args.config_dir, "python-fbas.cfg")
        if os.path.exists(config_candidate):
            args.config_file = config_candidate

    # Load configuration file first (if it exists)
    try:
        load_config_file(args.config_file)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)

    # CLI arguments override config file settings
    config_kwargs = {}

    # Only set CLI values that are not defaults (to preserve config file
    # values)
    if args.output_problem is not None:
        config_kwargs['output'] = args.output_problem
    if args.cardinality_encoding != 'totalizer':  # default value
        config_kwargs['card_encoding'] = args.cardinality_encoding
    if args.group_by is not None:
        config_kwargs['group_by'] = args.group_by
    if args.sat_solver != 'cryptominisat5':  # default value
        config_kwargs['sat_solver'] = args.sat_solver
    if args.max_sat_algo != 'LSU':  # default value
        config_kwargs['max_sat_algo'] = args.max_sat_algo
    if args.validator_display != 'both':  # default value
        config_kwargs['validator_display'] = args.validator_display
    if args.fbas and (args.fbas.startswith('http://')
                      or args.fbas.startswith('https://')):
        config_kwargs['stellar_data_url'] = args.fbas

    if config_kwargs:
        update_config(**config_kwargs)
    cfg = get_config()

    # Run commands that don't need FBAS data:
    if args.command in [
        'update-cache',
        'show-config',
        'show-generator-config',
        'show-sybil-detection-config',
        'random-sybil-attack-fbas',
    ]:
        args.func(args)
        sys.exit(0)

    # Validate configuration for commands that need it
    if cfg.card_encoding not in ['naive', 'totalizer']:
        logging.error("Error: Cardinality encoding must be either 'naive' or 'totalizer'")
        sys.exit(1)

    if cfg.sat_solver not in solvers:
        logging.error(f"Error: Solver must be one of {solvers}")
        sys.exit(1)

    if cfg.max_sat_algo not in ['LSU', 'RC2']:
        logging.error("Error: MaxSAT algorithm must be either 'LSU' or 'RC2'")
        sys.exit(1)

    # Validate that --update-cache is only used with URLs
    if args.update_cache:
        cfg = get_config()
        # Check if using URL (either from --fbas or default)
        using_url = False
        if args.fbas and (args.fbas.startswith('http://')
                          or args.fbas.startswith('https://')):
            using_url = True
        elif not args.fbas and (cfg.stellar_data_url.startswith('http://')
                                or cfg.stellar_data_url.startswith('https://')):
            using_url = True

        if not using_url:
            logging.error("Error: --update-cache can only be used with URLs")
            sys.exit(1)

    fbas = _load_fbas_graph(args)
    if cfg.group_by is not None:
        missing = [
            v for v in fbas.get_validators()
            if not fbas.vertice_attrs(v).get(cfg.group_by)
        ]
        if missing:
            unknown_label = fbas.group_unknown_label(cfg.group_by)
            logging.warning(
                "Some validators do not have the \"%s\" attribute; treating as \"%s\"",
                cfg.group_by,
                unknown_label,
            )
    if args.reachable_from:
        fbas = fbas.restrict_to_reachable(args.reachable_from)

    args.func(args, fbas)
    sys.exit(0)


if __name__ == "__main__":
    main()
