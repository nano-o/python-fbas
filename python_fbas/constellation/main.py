"""
For compute-clusters, you must first run 'make' in the 'brute-force-search/' directory and then make
sure that the executable 'optimal_cluster_assignment' is in your PATH.
"""

import argparse
import logging
import sys
import subprocess
from python_fbas.constellation.constellation import *
import python_fbas.constellation.config as config

# TODO allow taking a fbas in stellarbeat format as input

def main():
    parser = argparse.ArgumentParser(description="Constellation CLI")
    # specify log level with --log-level, with default WARNING:
    parser.add_argument('--log-level', default='WARNING', help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    clusters_parser = subparsers.add_parser('compute-clusters', help="compute an assignment of organizations to clusters")
    clusters_parser.add_argument('--thresholds', nargs='+', type=int, required=True, help="List of the form 'm1 t1 m2 t2 ... mn tn' where each ti is a quorum-set threshold and mi is the number of organizations that have this threshold")
    clusters_parser.add_argument('--min-cluster-size', type=int, default=1, help="Minimum size of a cluster (default: 1 organization)")
    clusters_parser.add_argument('--max-num-clusters', type=int, help="Maximum number of clusters (default: number of organizations)")

    compute_overlay_parser = subparsers.add_parser('compute-overlay', help="compute an overlay graph using the Constellation algorithm")
    compute_overlay_parser.add_argument('--fbas', type=str, required=True, help="Path to a JSON file describing a single-universe, regular FBAS. This must be a dict mapping orgs to integer thresholds.")
    compute_overlay_parser.add_argument('--output', type=str, required=True, help="Path to a JSON file where the overlay graph will be saved")
    compute_overlay_parser.add_argument('--min-cluster-size', type=int, default=1, help="Minimum size of a cluster (default: 1 organization)")
    compute_overlay_parser.add_argument('--max-num-clusters', type=int, help="Maximum number of clusters (default: number of organizations)")

    generate_parser = subparsers.add_parser('generate-random', help="generate a random single-universe FBA system")
    generate_parser.add_argument('--num-orgs', type=int, required=True, help="Number of organizations")
    generate_parser.add_argument('--min-threshold', type=int, default=1, help="Minimum quorum-set threshold (default: 1)")
    generate_parser.add_argument('--max-threshold', type=int, default=0, help="Maximum quorum-set threshold (default: number of orgs)")
    generate_parser.add_argument('--num-thresholds', type=int, default=0, help="Number of different quorum-set thresholds (default: no limit)")
    
    args = parser.parse_args()

    # set log level:
    debug_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if args.log_level not in debug_levels:
        logging.error("Log level must be one of %s", debug_levels)
        sys.exit(1)
    logging.getLogger().setLevel(args.log_level)

    if args.command == 'compute-clusters':
        if len(args.thresholds) % 2 != 0 \
            or len(args.thresholds) == 0 \
            or any(not isinstance(args.thresholds[i], int) or args.thresholds[i] <= 0 
                   for i in range(len(args.thresholds))) \
            or any(args.thresholds[1::2].count(args.thresholds[i]) > 1 for i in range(1,len(args.thresholds),2)):
            logging.error("Quorum-set thresholds and their multiplicity must be provided as a list 'm1 t1 m2 t2 ... mn tn' of strictly positive integer with no duplicate thresholds.")
            sys.exit(1)
        config.max_num_clusters = args.max_num_clusters
        config.min_cluster_size = args.min_cluster_size
        # create a single-universe fbas from the command line arguments:
        single_univ_fbas = {}
        i = 0
        for j in range(0,len(args.thresholds),2):
            for _ in range(args.thresholds[j]):
                i += 1
                single_univ_fbas[f"O_{i}"] = args.thresholds[j+1]
        # now compute the clusters:
        clusters = compute_clusters(single_univ_fbas)
        print(f"There are {len(clusters)} clusters of size {[len(c) for c in clusters]}")
        print(clusters)
    elif args.command == 'compute-overlay':
        config.max_num_clusters = args.max_num_clusters
        config.min_cluster_size = args.min_cluster_size
        with open(args.fbas, 'r', encoding='utf-8') as f:
            fbas = json.load(f)
        overlay:nx.Graph = constellation_overlay(fbas)
        # save the overlay graph to an the output file in JSON format:
        graph_data = nx.node_link_data(overlay)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=4)
    elif args.command == 'generate-random':
        if args.max_threshold == 0:
            args.max_threshold = args.num_orgs
        fbas = random_single_universe_regular_fbas(args.num_orgs, args.min_threshold, args.max_threshold, args.num_thresholds if args.num_thresholds > 0 else None)
        print(json.dumps(fbas, indent=4))
    
    # print help:
    elif args.command is None:
        parser.print_help()

if __name__ == "__main__":
    main()