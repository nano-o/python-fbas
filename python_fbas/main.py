"""Main CLI for the FBAS analysis tool."""

import argparse
import json
import logging
from typing import Iterable

from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import (
    find_disjoint_quorums,
    find_minimal_blocking_set,
    find_minimal_splitting_set,
    find_min_quorum,
    min_history_loss_critical_set,
    solvers,
    top_tier,
)
from python_fbas.pubnet_data import get_validators
import python_fbas.config as config


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_fbas(args: argparse.Namespace) -> FBASGraph:
    if args.fbas == "pubnet":
        return FBASGraph.from_json(get_validators())
    return FBASGraph.from_json(_load_json(args.fbas))


def _prepare_fbas(args: argparse.Namespace) -> FBASGraph:
    fbas = _load_fbas(args)
    if config.group_by is not None and not all(
        config.group_by in fbas.vertice_attrs(v) for v in fbas.validators
    ):
        raise ValueError(
            f"Some validators do not have the \"{config.group_by}\" attribute"
        )
    if args.reachable_from:
        fbas = fbas.restrict_to_reachable(args.reachable_from)
    return fbas


def _names(fbas: FBASGraph, vs: Iterable[str]) -> list[str]:
    return [fbas.with_name(v) for v in vs]


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_update_cache(_: argparse.Namespace) -> int:
    get_validators(update=True)
    return 0


def cmd_check_intersection(args: argparse.Namespace) -> int:
    if args.group_by:
        logging.error("--group-by does not make sense with check-intersection")
        return 1
    fbas = _prepare_fbas(args)
    if args.fast:
        result = fbas.fast_intersection_check()
        print(f"Intersection-check result: {result}")
        return 0
    result = find_disjoint_quorums(fbas)
    if result:
        print(
            f"Disjoint quorums: {_names(fbas, result[0])}\n and {_names(fbas, result[1])}"
        )
    else:
        print("No disjoint quorums found")
    return 0


def cmd_min_splitting_set(args: argparse.Namespace) -> int:
    fbas = _prepare_fbas(args)
    result = find_minimal_splitting_set(fbas)
    if not result:
        print("No splitting set found")
        return 0
    q1, q2 = result[1], result[2]
    split = result[0]
    print(f"Minimal splitting-set cardinality is: {len(split)}")
    example = _names(fbas, split) if config.group_by is None else split
    print(
        f"Example:\n{example}\nsplits quorums\n{_names(fbas, q1)}\nand\n{_names(fbas, q2)}"
    )
    return 0


def cmd_min_blocking_set(args: argparse.Namespace) -> int:
    fbas = _prepare_fbas(args)
    result = find_minimal_blocking_set(fbas)
    if not result:
        print("No blocking set found")
        return 0
    example = _names(fbas, result) if config.group_by is None else result
    print(f"Minimal blocking-set cardinality is: {len(result)}")
    print(f"Example:\n{example}")
    return 0


def cmd_history_loss(args: argparse.Namespace) -> int:
    if args.group_by:
        logging.error("--group-by does not make sense with history-loss")
        return 1
    fbas = _prepare_fbas(args)
    result = min_history_loss_critical_set(fbas)
    critical, quorum = result
    print(
        f"Minimal history-loss critical set cardinality is: {len(critical)}"
    )
    print(f"Example min critical set:\n{_names(fbas, critical)}")
    print(f"Corresponding history-less quorum:\n {_names(fbas, quorum)}")
    return 0


def cmd_min_quorum(args: argparse.Namespace) -> int:
    fbas = _prepare_fbas(args)
    result = find_min_quorum(fbas)
    print(f"Example min quorum:\n{_names(fbas, result)}")
    return 0


def cmd_top_tier(args: argparse.Namespace) -> int:
    fbas = _prepare_fbas(args)
    result = top_tier(fbas)
    print(f"Top tier: {_names(fbas, result)}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing and main entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBAS analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--fbas",
        default="pubnet",
        help=(
            "Where to find the description of the FBAS to analyze. "
            "Must be 'pubnet' or a path to a JSON file."
        ),
    )
    parser.add_argument(
        "--reachable-from",
        default=None,
        help="Restrict the FBAS to what's reachable from the provided validator",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help=(
            "Group by the provided field (e.g. min-splitting-set with "
            "--group-by=homeDomain will compute the minimum number of home "
            "domains to corrupt to create disjoint quorums)"
        ),
    )
    parser.add_argument(
        "--cardinality-encoding",
        default="totalizer",
        choices=["naive", "totalizer"],
        help="Cardinality encoding",
    )
    parser.add_argument(
        "--sat-solver",
        default="cryptominisat5",
        choices=solvers,
        help="SAT solver to use. See pysat documentation for more information.",
    )
    parser.add_argument(
        "--max-sat-algo",
        default="LSU",
        choices=["LSU", "RC2"],
        help="MaxSAT algorithm to use",
    )
    parser.add_argument(
        "--output-problem",
        default=None,
        help="Write the constraint-satisfaction problem to the provided path",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sp = subparsers.add_parser(
        "update-cache", help=f"Update data form {config.stellar_data_url}"
    )
    sp.set_defaults(func=cmd_update_cache)

    sp = subparsers.add_parser(
        "check-intersection",
        help="Check that the FBAS is intertwined (i.e. whether all quorums intersect)",
    )
    sp.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use the fast heuristic (which does not use a SAT solver and only "
            "returns true, meaning all quorums intersect, or unknown)"
        ),
    )
    sp.set_defaults(func=cmd_check_intersection)

    sp = subparsers.add_parser(
        "min-splitting-set", help="Find minimal-cardinality splitting set"
    )
    sp.set_defaults(func=cmd_min_splitting_set)

    sp = subparsers.add_parser(
        "min-blocking-set", help="Find minimal-cardinality blocking set"
    )
    sp.set_defaults(func=cmd_min_blocking_set)

    sp = subparsers.add_parser(
        "history-loss",
        help=(
            "Find a minimal-cardinality set of validators such that, should "
            "they stop publishing valid history, a full quorum could advance "
            "without publishing valid history (in which case history may be lost)"
        ),
    )
    sp.set_defaults(func=cmd_history_loss)

    sp = subparsers.add_parser("min-quorum", help="Find minimal-cardinality quorum")
    sp.set_defaults(func=cmd_min_quorum)

    sp = subparsers.add_parser("top-tier", help="Find the top tier of the FBAS")
    sp.set_defaults(func=cmd_top_tier)

    return parser.parse_args(argv)


def configure(args: argparse.Namespace) -> None:
    config.output = args.output_problem
    config.card_encoding = args.cardinality_encoding
    config.group_by = args.group_by
    config.sat_solver = args.sat_solver
    config.max_sat_algo = args.max_sat_algo
    logging.getLogger().setLevel(args.log_level)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
