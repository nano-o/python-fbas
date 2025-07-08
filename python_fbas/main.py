"""
Main CLI for the FBAS analysis tool
"""

from collections.abc import Collection
import json
import argparse
import logging
import sys
from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import (
    find_disjoint_quorums,
    find_minimal_splitting_set, find_minimal_blocking_set,
    min_history_loss_critical_set, find_min_quorum, top_tier, max_scc
)
from python_fbas.pubnet_data import get_pubnet_config
from python_fbas.solver import solvers
from python_fbas.config import update as update_config, get as get_config


def _load_json_from_file(validators_file):
    with open(validators_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_fbas_graph(args) -> FBASGraph:
    if args.fbas == 'pubnet':
        return FBASGraph.from_json(get_pubnet_config())
    return FBASGraph.from_json(_load_json_from_file(args.fbas))


def _with_names(fbas: FBASGraph, vs: Collection[str]) -> list[str]:
    return [fbas.with_name(v) for v in vs]


def _command_update_cache(_args):
    get_pubnet_config(update=True)
    print("Cache updated.")


def _command_check_intersection(args, fbas: FBASGraph):
    cfg = get_config()
    if cfg.group_by:
        logging.error(
            "--group-by does not make sense with check-intersection")
        sys.exit(1)
    if args.fast:
        result = fbas.fast_intersection_check()
        print(f"Intersection-check result: {result}")
    else:
        result = find_disjoint_quorums(fbas)
        if result:
            print(
                f"Disjoint quorums: {_with_names(fbas, result.quorum_a)}\n and {_with_names(fbas, result.quorum_b)}")
        else:
            print("No disjoint quorums found")


def _command_min_splitting_set(_args, fbas: FBASGraph):
    cfg = get_config()
    result = find_minimal_splitting_set(fbas)
    if not result:
        print("No splitting set found")
        return
    print(f"Minimal splitting-set cardinality is: {len(result.splitting_set)}")
    print(
        f"Example:\n{_with_names(fbas, result.splitting_set) if not cfg.group_by else result.splitting_set}\nsplits quorums\n{_with_names(fbas, result.quorum_a)}\nand\n{_with_names(fbas, result.quorum_b)}")


def _command_min_blocking_set(_args, fbas: FBASGraph):
    cfg = get_config()
    result = find_minimal_blocking_set(fbas)
    if not result:
        print("No blocking set found")
        return
    print(f"Minimal blocking-set cardinality is: {len(result)}")
    print(
        f"Example:\n{_with_names(fbas, result) if not cfg.group_by else result}")


def _command_history_loss(_args, fbas: FBASGraph):
    cfg = get_config()
    if cfg.group_by:
        logging.error("--group-by does not make sense for the history-loss command")
        sys.exit(1)
    result = min_history_loss_critical_set(fbas)
    print(
        f"Minimal history-loss critical set cardinality is: {len(result.min_critical_set)}")
    print(f"Example min critical set:\n{_with_names(fbas, result.min_critical_set)}")
    print(
        f"Corresponding history-less quorum:\n {_with_names(fbas, result.quorum)}")


def _command_min_quorum(_args, fbas: FBASGraph):
    result = find_min_quorum(fbas)
    print(f"Example min quorum:\n{_with_names(fbas, result)}")


def _command_top_tier(_args, fbas: FBASGraph):
    result = top_tier(fbas)
    print(f"Top tier: {_with_names(fbas, result)}")


def _command_max_scc(_args, fbas: FBASGraph):
    cfg = get_config()
    if cfg.group_by:
        logging.error("--group-by does not make sense for the max-scc command")
        sys.exit(1)
    result = max_scc(fbas)
    print(f"Maximal SCC with a quorum: {_with_names(fbas, result)}")


def main():
    parser = argparse.ArgumentParser(description="FBAS analysis CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument(
        '--log-level',
        default='WARNING',
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # specify a data source:
    parser.add_argument(
        '--fbas',
        default='pubnet',
        help="Where to find the description of the FBAS to analyze. Must be 'pubnet' or a path to a JSON file. The pubnet data source is set in config.py")
    parser.add_argument(
        '--reachable-from',
        default=None,
        help="Restrict the FBAS to what's reachable from the provided validator")
    parser.add_argument(
        '--group-by',
        default=None,
        help="Group by the provided field (e.g. min-splitting-set with --group-by=homeDomain will compute the minimum number of home domains to corrupt to create disjoint quorums)")

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
        help=f"Update data from {get_config().stellar_data_url}")
    parser_update_cache.set_defaults(func=_command_update_cache)

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
    parser_top_tier.set_defaults(func=_command_top_tier)

    parser_max_scc = subparsers.add_parser(
        'max-scc', help="Find a maximal strongly-connected component of the FBAS that contains a quorum")
    parser_max_scc.set_defaults(func=_command_max_scc)

    args = parser.parse_args()

    update_config(output=args.output_problem,
                  card_encoding=args.cardinality_encoding,
                  group_by=args.group_by,
                  sat_solver=args.sat_solver,
                  max_sat_algo=args.max_sat_algo)
    cfg = get_config()

    if cfg.card_encoding not in ['naive', 'totalizer']:
        logging.error(
            "Cardinality encoding must be either 'naive' or 'totalizer'")
        sys.exit(1)

    if cfg.sat_solver not in solvers:
        logging.error("Solver must be one of %s", solvers)
        sys.exit(1)

    if cfg.max_sat_algo not in ['LSU', 'RC2']:
        logging.error("MaxSAT algorithm must be either 'LSU' or 'RC2'")
        sys.exit(1)

    debug_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if args.log_level not in debug_levels:
        logging.error("Log level must be one of %s", debug_levels)
        sys.exit(1)
    logging.getLogger().setLevel(args.log_level)

    # Run commands:
    if args.command == 'update-cache':
        args.func(args)
        sys.exit(0)

    fbas = _load_fbas_graph(args)
    if cfg.group_by is not None and not all(
            cfg.group_by in fbas.vertice_attrs(v) for v in fbas.validators):
        raise ValueError(
            f"Some validators do not have the \"{cfg.group_by}\" attribute")
    if args.reachable_from:
        fbas = fbas.restrict_to_reachable(args.reachable_from)

    args.func(args, fbas)
    sys.exit(0)


if __name__ == "__main__":
    main()
