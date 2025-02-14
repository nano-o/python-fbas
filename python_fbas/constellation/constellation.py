import json
import logging
import subprocess
import random
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, product
from typing import Optional, Tuple
import networkx as nx
from python_fbas.fbas_graph import FBASGraph
import python_fbas.constellation.config as config

def fbas_graph_to_single_universe_regular(fbas_graph:FBASGraph) -> Tuple[dict[str,int], dict[str,int]]:
    """
    Convert a FBASGraph to a single-universe regular FBAS, along with the number of validators of each organization.

    TODO this should return a dict mapping to pairs no?
    """
    thresholds:dict[str,int] = {}
    num_validators:dict[str,int] = {}
    # TODO validate we indeed have a regular FBAS, and if not single-universe then make it so.
    for v, attrs in fbas_graph.vertices(data=True):
        if v in fbas_graph.validators:
            # assert v has homeDomain:
            assert 'homeDomain' in attrs
            org = attrs['homeDomain']
            thresholds[org] = fbas_graph.qsets[fbas_graph.qset_vertex_of(v)].threshold
            if org in num_validators:
                num_validators[org] += 1
            else:
                num_validators[org] = 1
    return thresholds, num_validators

def single_universe_to_regular(fbas: dict[str,int]) -> dict[str,tuple[int,list[str]]]:
    """
    Convert a single-universe regular FBAS to a regular FBAS.
    """
    orgs = list(fbas.keys())
    return {org: (fbas[org], orgs) for org in fbas}

def random_single_universe_regular_fbas(n:int, low:int, high:int, num_thresholds:Optional[int] = None) -> dict:
    """
    Generate a random single-universe regular FBAS with n organizations and thresholds between low
    and high. This is just a dict from organization names to thresholds.

    Optionally limit the number of different threshold that appear.
    """
    if num_thresholds is not None:
        num_thresholds = min(high-low, num_thresholds)
        thresholds = random.sample(range(low, high+1), num_thresholds)
        return {f"O_{i}": random.choice(thresholds) for i in range(1, n+1)}
    else:
        return {f"O_{i}": random.randint(low, high) for i in range(1, n+1)}

def single_universe_regular_fbas_to_fbas_graph(fbas:dict[str,int]) -> FBASGraph:
    """
    Convert a single-universe regular FBAS to a FBASGraph.
    """
    assert isinstance(fbas, dict)
    for org in fbas:
        assert isinstance(org, str)
        assert isinstance(fbas[org], int)
        assert fbas[org] > 0 and fbas[org] <= len(fbas)
    fbas_graph = FBASGraph()
    # Each org has 3 validators and inner threshold 2:
    inner_qsets = [{'threshold': 2, 'validators': [f'{org}_0', f'{org}_1', f'{org}_2'], 'innerQuorumSets': []} for org in fbas]
    for org in fbas:
        qset = {'threshold': fbas[org], 'validators': [], 'innerQuorumSets': inner_qsets}
        attrs = {'homeDomain': org}
        for i in range(0, 3):
            fbas_graph.update_validator(f'{org}_{i}', qset, attrs)
    return fbas_graph

def load_survey_graph(file_name) -> nx.Graph:
    """
    Load the overlay graph from Stellar survey data.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    g = nx.Graph()
    for v, peers in json_data.items():
        inbound = set(peers['inboundPeers'].keys())
        outbound = set(peers['outboundPeers'].keys())
        for peer in inbound | outbound:
            g.add_edge(v, peer)
    return g

def regular_fbas_to_fbas_graph(regular_fbas) -> FBASGraph:
    """
    Convert a regular FBAS to a FBASGraph. A regular FBAS consists of a set of organizations where
    each organization O requires agreement from a threshold t_O among a set of organizations S_O.
    """
    assert isinstance(regular_fbas, dict)
    for org in regular_fbas:
        assert isinstance(org, str)
        # org must be map to a pair (threshold, list of organizations):
        assert isinstance(regular_fbas[org], tuple)
        assert isinstance(regular_fbas[org][0], int)
        assert regular_fbas[org][0] > 0
        assert isinstance(regular_fbas[org][1], list)
        for o in regular_fbas[org][1]:
            assert isinstance(o, str)
            assert o in regular_fbas

    # now build the FBASGraph
    def org_inner_qset(o):
        return {'threshold': 2, 'validators': [f'{o}_0', f'{o}_1', f'{o}_2'], 'innerQuorumSets': []}
    fbas_graph = FBASGraph()
    for org in regular_fbas:
        match regular_fbas[org]:
            case (threshold, orgs):
                qset = {'threshold': threshold, 'validators': [],'innerQuorumSets': [org_inner_qset(o) for o in orgs]}
                for n in range(0, 3):
                    fbas_graph.update_validator(f'{org}_{n}', qset)
    return fbas_graph

def parse_output(output:str) -> list[dict[int,int]]:
    """
    Parse the output of the optimal partitioning algorithm.

    A partition is represented by a string; for example, "1.1.2.2.2.3.5|1.4.4|2.2.3" means [{1:2, 2:3, 3:1, 5:1}, {1:1, 4:2}, {2:2, 3:1}].
    In the output of the C program, the optimal partition appears on the last but 2 line.
    """
    # get the last but 2 line:
    lines = output.splitlines()
    partition = lines[-2]
    partitions = partition.split('|')
    result = []
    for p in partitions:
        d:dict[int,int] = defaultdict(int)
        for x in p.split('.'):
            d[int(x)] += 1
        result.append(d)
    return result

def compute_clusters(fbas:dict[str,int]) -> list[set[str]]:
    """
    Determines the Constellation clusters by calling the C implementation of the optimal-partitioning algorithm.
    The command 'optimal_cluster_assignment' must be in the PATH.

    This only uses the threshold of each organization, not its universe (which is implicitely assumed to be all organizations).
    """
    n_orgs = len(fbas.keys())
    threshold_multiplicity:dict[int,int] = defaultdict(int)
    for org,t in fbas.items():
            if isinstance(t, int):
                threshold_multiplicity[n_orgs - t + 1] += 1
            else:
                raise ValueError("Expected an integer threshold")
    # build the command-line arguments:
    arg_pairs = [[threshold_multiplicity[t], t] for t in threshold_multiplicity.keys()]
    args = [x for sublist in arg_pairs for x in sublist] # flatten the list
    # # if there are more than n_limit organizations, limit the mimimum cluster size to n_orgs/size_denom and the max number of clusters to max_clusters
    # n_limit = 12
    # size_denom = 5
    # min_cluster_size = int(n_orgs/size_denom)+1 if n_orgs > n_limit else 1
    # # limit the number of clusters to 4:
    # max_clusters = 4 if n_orgs > n_limit else n_orgs
    args = args + [config.min_cluster_size, config.max_num_clusters if config.max_num_clusters else n_orgs]
    # obtain the optimal partition:
    logging.debug("calling optimal_cluster_assignment with args: %s", args)
    output = subprocess.run(['optimal_cluster_assignment'] + [str(x) for x in args],
                            capture_output=True, text=True, check=True, timeout=config.timeout) # exception on timeout
    partition = parse_output(output.stdout) # TODO error handling
    # now assign organizations to the clusters
    threshold_map:dict[int,list[str]] = {} # map each threshold to the set of organizations that have it
    for org,t in fbas.items():
        t = n_orgs - t + 1
        if t not in threshold_map:
            threshold_map[t] = []
        threshold_map[t].append(org)
    # sort the blocking thresholds in decreasing order:
    index_of_threshold = {t:i for i,t in enumerate(sorted(threshold_multiplicity.keys(), reverse=True))}
    clusters:list[set[str]] = [set() for _ in partition] # start with empty cluters
    for t, orgs in threshold_map.items():
        for i, part in enumerate(partition):
            n = part.get(index_of_threshold[t]+1, 0)
            clusters[i] |= set(orgs[:n])
            orgs = orgs[n:]

    logging.info("computed %s clusters", len(clusters))
    return clusters

def clusters_to_overlay(clusters:list[set[str]], num_validators:Optional[dict[str,int]]=None) -> nx.Graph:
    """
    Given a list of clusters, return the Constellation overlay graph.
    The number of validators of each organization can be specified in the num_validators dict.
    We assume by default that each organization has 3 validators.
    Regardless of the its number of validators N, we assume each organization has an inner threshold of int(N/2)+1.

    Easy case: each organization O has 3 nodes O_0, O_1, and O_2 that form a fully connected graph.
    If two organizations are in different clusters, then each node in the first organization is connected to each node in the second organization.
    If two organizations O and O' are in the same cluster, then O_i is connected to O'_{i+1 mod 3} and O'{i+2 mod 3}.

    In the general case, we still form a fully connected graph among each organization.
    If two organizations are in different clusters, then each node in the first organization is connected to each node in the second organization.
    If two organizations O and O' are in the same cluster, then O_i is connected to O'_{i+1 mod 3} and O'{i+2 mod 3}.
    """
    # by default, assign 3 validators to each organization
    num_validators = num_validators or {org: 3 for cluster in clusters for org in cluster}
    g = nx.Graph()
    # first, connect the org's nodes in a complete graph
    for org in set.union(*clusters):
        for i, j in combinations(range(num_validators[org]), 2):
            g.add_edge(f'{org}_{i}', f'{org}_{j}')
    # now connect the org's nodes to the nodes of other organizations in the same cluster:
    for c in clusters:
        for org, other_org in combinations(c, 2):
            n = num_validators[org]
            half = int(n/2)+1
            n_other = num_validators[other_org]
            half_other = int(n_other/2)+1
            for i in range(n):
                for j in range(half_other):
                    g.add_edge(f'{org}_{i}', f'{other_org}_{(i+j)%n_other}')
            for j in range(n_other):
                for i in range(half):
                    g.add_edge(f'{org}_{(j-i)%n}', f'{other_org}_{j}')
    # now create the inter-cluster edges:
    for c1, c2 in combinations(clusters, 2):
        max_c,min_c = (list(c1),list(c2)) if len(c1) > len(c2) else (list(c2),list(c1))
        for n, org in enumerate(max_c):
            other = min_c[n%len(min_c)]
            for i,j in product(range(num_validators[org]), range(num_validators[other])):
                g.add_edge(f'{org}_{i}', f'{other}_{j}')
    logging.info("The average degree of the overlay graph is %s", sum([d for n,d in g.degree()])/len(g.nodes()))
    return g

def constellation_overlay(fbas:dict[str,int], num_validators:Optional[dict[str,int]]=None) -> nx.Graph:
    """
    Given a regular FBAS, return the Constellation overlay graph.
    """
    # first we transform the regular fbas into a single-universe regular fbas:
    clusters = compute_clusters(fbas)
    return clusters_to_overlay(clusters, num_validators=num_validators)

def constellation_overlay_of_fbas_graph(fbas_graph:FBASGraph) -> nx.Graph:
    """
    Given a FBASGraph, return the Constellation overlay graph.
    """
    fbas, num_validators = fbas_graph_to_single_universe_regular(fbas_graph)
    return constellation_overlay(fbas, num_validators=num_validators)


def reduce_diameter_to_2(g):
    """ Randomly adds edges to reduce the graph's diameter to at most 2. """
    # Check initial diameter
    if nx.diameter(g) <= 2:
        return g
    # Find shortest paths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(g))
    # Identify node pairs at distance >= 3
    distant_pairs = [(u, v) for u in shortest_paths for v in shortest_paths[u] if shortest_paths[u][v] >= 3]
    # remove permutations:
    distant_pairs = set(tuple(sorted(pair)) for pair in distant_pairs)
    # sort by decreasing shortest path:
    distant_pairs = sorted(distant_pairs, key=lambda x: shortest_paths[x[0]][x[1]], reverse=True)
    # Add edges iteratively until diameter is 2
    for u, v in distant_pairs:
        g.add_edge(u, v)
        if nx.diameter(g) <= 2:
            break  # Stop early if diameter is already reduced
    return g

def greedy_overlay(fbas:dict[str,int]) -> nx.Graph:
    """
    Given a single-universe regular FBAS, compute an overlay using the greedy strategy.
    """

    # first we create a graph over orgs
    orgs = list(fbas.keys())
    n_orgs = len(orgs)
    def req(org) -> int: # required number of connections (including to self)
        return n_orgs - fbas[org] + 1
    # sort the orgs in descending number of required connections:
    sorted_orgs = sorted(fbas.keys(), key=req, reverse=True)
    orgs_graph = nx.Graph()
    for i, org in enumerate(sorted_orgs):
        # connect org i to orgs i+1 to i+req(org), and if i+req(org) > n_orgs, then pick
        # i+req(org)-n_orgs random additional orgs (not yet connected to) to connect to.
        for j in range(i+1, i+req(org)): # only i+req(org)-1 because self counts as a connection
            if j < n_orgs:
                orgs_graph.add_edge(org, sorted_orgs[j])
            else:
                # skip j if we already have enough connections:
                if len(list(orgs_graph.neighbors(org))) >= j-i:
                    continue
                # pick a random org not yet connected to:
                not_connected = set(sorted_orgs) - (set(orgs_graph.neighbors(org)) | {org})
                orgs_graph.add_edge(org, random.choice(list(not_connected)))

    # next we make the node to node connections
    g = nx.Graph()
    for o1, o2 in combinations(orgs, 2):
        for i in range(0, 3):
            for j in range(0,3):
                g.add_edge(f'{o1}_{i}', f'{o2}_{(i+j)%3}')
    # finally, we reduce the diameter to 2
    # NOTE seems diameter is already 2 in most cases
    g = reduce_diameter_to_2(g)
    return g
