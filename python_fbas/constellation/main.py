"""
For compute-clusters, you must first run 'make' in the 'brute-force-search/' directory and then make
sure that the executable 'optimal_cluster_assignment' is in your PATH.
"""
import os
import argparse
import logging
import sys
import subprocess
from python_fbas.constellation.constellation import *
import python_fbas.constellation.config as config

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
    
    args = parser.parse_args()

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
        single_univ_fbas = {}
        i = 0
        for j in range(0,len(args.thresholds),2):
            for _ in range(args.thresholds[j]):
                i += 1
                single_univ_fbas[f"O_{i}"] = args.thresholds[j+1]
        clusters = compute_clusters(single_univ_fbas)
        print(f"There are {len(clusters)} clusters of size {[len(c) for c in clusters]}")
        print(clusters)
    elif args.command == 'compute-overlay':
        os.chdir("python_fbas/constellation")
        with open(args.fbas, 'r', encoding='utf-8') as f:
            fbas = json.load(f)
        overlay:nx.Graph = constellation_overlay(fbas)
        # print the average degree:
        avg_degree = sum([d for n,d in overlay.degree()])/len(overlay.nodes())
        print(f"Average degree: {avg_degree}")
        # save the overlay graph to an the output file in JSON format:
        # instead we print overlay data for docker img
        graph_data = nx.node_link_data(overlay)
        print(f"Graph data: {graph_data}")
        # with open(args.output, 'w', encoding='utf-8') as f:
        #     json.dump(graph_data, f, indent=4)
    
    # print help:
    elif args.command is None:
        parser.print_help()

if __name__ == "__main__":
    main()