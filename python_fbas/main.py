"""
Main CLI for the FBAS analysis tool
"""

from collections.abc import Collection
import json
import argparse
import logging
import sys
from typing import Any, Dict, List
from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import (
    find_disjoint_quorums,
    find_minimal_splitting_set, find_minimal_blocking_set,
    min_history_loss_critical_set, find_min_quorum, top_tier, max_scc
)
from python_fbas.pubnet_data import get_pubnet_config
from python_fbas.solver import solvers
from python_fbas.config import update as update_config, get as get_config, load_config_file, to_yaml
from python_fbas.serialization import deserialize, serialize


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


def _command_to_json(args: Any, fbas: FBASGraph) -> None:
    """Convert the loaded FBAS to JSON format and print to stdout."""
    if args.format == 'python-fbas':
        print(serialize(fbas, format='python-fbas'))
    elif args.format == 'stellarbeat':
        print(serialize(fbas, format='stellarbeat'))
    else:
        logging.error(f"Error: Unknown format '{args.format}'. Must be 'python-fbas' or 'stellarbeat'")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="FBAS analysis CLI")
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

    parser_to_json = subparsers.add_parser(
        'to-json', help="Convert the loaded FBAS to JSON format and print to stdout")
    parser_to_json.add_argument(
        '--format',
        default='python-fbas',
        choices=['python-fbas', 'stellarbeat'],
        help="Output format: 'python-fbas' (default) or 'stellarbeat'")
    parser_to_json.set_defaults(func=_command_to_json)

    args = parser.parse_args()

    # Set log level early
    debug_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if args.log_level not in debug_levels:
        print(f"Error: Log level must be one of {debug_levels}", file=sys.stderr)
        sys.exit(1)
    logging.getLogger().setLevel(args.log_level)

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
    if args.command in ['update-cache', 'show-config']:
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
