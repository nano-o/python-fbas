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
    min_history_loss_critical_set, find_min_quorum, top_tier
    )
from python_fbas.pubnet_data import get_validators
from python_fbas.solver import solvers
import python_fbas.config as config


def _load_json_from_file(validators_file):
    with open(validators_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_fbas_graph(args) -> FBASGraph:
    if args.fbas == 'pubnet':
        return FBASGraph.from_json(get_validators())
    return FBASGraph.from_json(_load_json_from_file(args.fbas))


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
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # Command for updating cached data
    subparsers.add_parser(
        'update-cache',
        help=f"Update data form {config.stellar_data_url}")

    # Command for checking intersection
    parser_is_intertwined = subparsers.add_parser(
        'check-intersection',
        help="Check that the FBAS is intertwined (i.e. whether all quorums intersect)")
    parser_is_intertwined.add_argument(
        '--fast',
        action='store_true',
        help="Use the fast heuristic (which does not use a SAT solver and only returns true, meaning all quorums intersect, or unknown)")

    # Command for minimum splitting set
    subparsers.add_parser('min-splitting-set',
                          help="Find minimal-cardinality splitting set")
    subparsers.add_parser('min-blocking-set',
                          help="Find minimal-cardinality blocking set")
    subparsers.add_parser(
        'history-loss',
        help="Find a minimal-cardinality set of validators such that, should they stop publishing valid history, would allow a full quorum to get ahead without publishing valid history (in which case history may be lost)")

    subparsers.add_parser('min-quorum', help="Find minimal-cardinality quorum")

    subparsers.add_parser('top-tier', help="Find the top tier of the FBAS")

    args = parser.parse_args()

    # set config:

    config.output = args.output_problem
    if args.cardinality_encoding not in ['naive', 'totalizer']:
        logging.error(
            "Cardinality encoding must be either 'naive' or 'totalizer'")
        sys.exit(1)
    config.card_encoding = args.cardinality_encoding

    config.group_by = args.group_by

    if args.sat_solver not in solvers:
        logging.error("Solver must be one of %s", solvers)
        sys.exit(1)
    config.sat_solver = args.sat_solver

    if args.max_sat_algo not in ['LSU', 'RC2']:
        logging.error("MaxSAT algorithm must be either 'LSU' or 'RC2'")
        sys.exit(1)
    config.max_sat_algo = args.max_sat_algo

    debug_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if args.log_level not in debug_levels:
        logging.error("Log level must be one of %s", debug_levels)
        sys.exit(1)
    logging.getLogger().setLevel(args.log_level)

    # Run commands:

    if args.command == 'update-cache':
        get_validators(update=True)
        sys.exit(0)

    fbas = _load_fbas_graph(args)
    if config.group_by is not None and not all(
            config.group_by in fbas.vertice_attrs(v) for v in fbas.validators):
        raise ValueError(
            f"Some validators do not have the \"{config.group_by}\" attribute")
    if args.reachable_from:
        fbas = fbas.restrict_to_reachable(args.reachable_from)

    def with_names(vs: Collection[str]) -> list[str]:
        return [fbas.with_name(v) for v in vs]

    if args.command == 'check-intersection':
        if args.group_by:
            logging.error(
                "--group-by does not make sense with check-intersection")
            exit(1)
        if args.fast:
            result = fbas.fast_intersection_check()
            print(f"Intersection-check result: {result}")
            sys.exit(0)
        else:
            result = find_disjoint_quorums(fbas)
            if result:
                print(
                    f"Disjoint quorums: {with_names(result[0])}\n and {with_names(result[1])}")
            else:
                print("No disjoint quorums found")
            sys.exit(0)
    elif args.command == 'min-splitting-set':
        result = find_minimal_splitting_set(fbas)
        if not result:
            print("No splitting set found")
            sys.exit(0)
        print(f"Minimal splitting-set cardinality is: {len(result[0])}")
        print(
            f"Example:\n{with_names(result[0]) if not config.group_by else result[0]}\nsplits quorums\n{with_names(result[1])}\nand\n{with_names(result[2])}")
        sys.exit(0)
    elif args.command == 'min-blocking-set':
        result = find_minimal_blocking_set(fbas)
        if not result:
            print("No blocking set found")
            sys.exit(0)
        print(f"Minimal blocking-set cardinality is: {len(result)}")
        print(
            f"Example:\n{with_names(result) if not config.group_by else result}")
        sys.exit(0)
    elif args.command == 'history-loss':
        if args.group_by:
            logging.error("--group-by does not make sense with history-loss")
            exit(1)
        result = min_history_loss_critical_set(fbas)
        print(
            f"Minimal history-loss critical set cardinality is: {len(result[0])}")
        print(f"Example min critical set:\n{with_names(result[0])}")
        print(f"Corresponding history-less quorum:\n {with_names(result[1])}")
        sys.exit(0)
    elif args.command == 'min-quorum':
        result = find_min_quorum(fbas)
        print(f"Example min quorum:\n{with_names(result)}")
        sys.exit(0)
    elif args.command == 'top-tier':
        result = top_tier(fbas)
        print(f"Top tier: {with_names(result)}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
