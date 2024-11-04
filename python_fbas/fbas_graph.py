"""
Federated Byzantine Agreement System (FBAS) represented as graphs.
"""

from typing import Any, Optional, Tuple
from collections.abc import Collection
import logging
from pprint import pformat
import networkx as nx
from python_fbas.utils import powerset

def freeze_qset(qset: dict) -> Tuple[int, frozenset]:
    """
    Expects a JSON-serializable quorum-set (in stellarbeat.io format) and returns a hashable version for use in collections.
    """
    match qset:
        case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
            threshold = int(t)
            members = frozenset(vs) | frozenset(freeze_qset(iq) for iq in iqs)
            assert 0 <= threshold <= len(members)
            assert not (threshold == 0 and len(members) > 0)
            return (threshold, members)
        case _:
            raise ValueError(f"Invalid qset: {qset}")


class FBASGraph:
    """
    A graph whose nodes are either validators or QSets.
    If n is a validator node, then it has at most one successor, which is a qset node. If it does not have a successor, then it's because its qset is unknown.
    If n is a qset node, then it has a threshold attribute and its successors are its validators and inner qsets.
    Each node has optional metadata attibutes.
    """
    graph: nx.DiGraph
    validators: set # only a subset of the nodes in the graph represent validators
    qset_count = 1
    qsets: dict[str, Tuple[int, frozenset]] # map qset nodes (str) to their data

    def __init__(self, graph=None, validators=None, qsets = None):
        if graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = graph
        if validators is None:
            self.validators = set()
        else:
            self.validators = validators
        if qsets is None:
            self.qsets = {}
        else:
            self.qsets = qsets

    def check_integrity(self):
        """Basic integrity checks"""
        if not self.validators <= self.graph.nodes():
            raise ValueError(f"Some validators are not in the graph: {self.validators - self.graph.nodes()}")
        for n, attrs in self.graph.nodes(data=True):
            if 'threshold' not in attrs:
                assert self.graph.out_degree(n) <= 1
            else:
                if attrs['threshold'] < 0 or attrs['threshold'] > self.graph.out_degree(n):
                    raise ValueError(f"Integrity check failed: threshold of {n} not in [0, out_degree={self.graph.out_degree(n)}]")
        for n in self.graph.nodes():
            if n in self.validators:
                if self.graph.out_degree(n) > 1:
                    raise ValueError(f"Integrity check failed: validator {n} has an out-degree greater than 1 ({self.graph.out_degree(n)})")
                if self.graph.out_degree(n) == 1:
                    assert self.qset_node_of(n) not in self.validators # TODO: do we want this invariant?
            if n in self.graph.successors(n):
                raise ValueError(f"Integrity check failed: node {n} has a self-loop")
        # check for loops of non-validator nodes:
        # nvg = self.graph.subgraph(n for n in self.graph.nodes() if n not in self.validators)
                
    def stats(self):
        """Compute some basic statistics"""
        def thresholds_distribution():
            return {t: sum(1 for _, attrs in self.graph.nodes(data=True) if 'threshold' in attrs and attrs['threshold'] == t)
                    for t in nx.get_node_attributes(self.graph, 'threshold').values()}
        return {
            'num_edges' : len(self.graph.edges()),
            'thresholds_distribution' : thresholds_distribution()
        }
        
    def add_validator(self, v:Any) -> None:
        """Add a validator to the graph."""
        self.graph.add_node(v)
        self.validators.add(v)

    def update_validator(self, v: Any, qset: Optional[dict] = None, attrs: Optional[dict] = None) -> None:
        """
        Add the validator v to the graph if it does not exist, using the supplied qset and attributes.
        Otherwise:
            - Update its attributes with attrs (existing attributes not in attrs remain unchanged).
            - Replace its outgoing edge with an edge to the given qset.
        Expects a qset, if given, in JSON-serializable stellarbeat.io format.
        """
        if attrs:
            self.graph.add_node(v, **attrs)
        else:
            self.graph.add_node(v)
        if qset:
            fqs = self.add_qset(qset)
            out_edges = list(self.graph.out_edges(v))
            self.graph.remove_edges_from(out_edges)
            self.graph.add_edge(v, fqs)
        self.validators.add(v)
    
    def add_qset(self, qset: dict) -> str:
        """
        Takes a qset as a JSON-serializable dict in stellarbeat.io format.
        Returns the qset if it already exists, otherwise adds it to the graph.
        """
        match qset:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
                fqs = freeze_qset(qset)
                if fqs in self.qsets.values():
                    return next(k for k,v in self.qsets.items() if v == fqs)
                iqs_nodes = [self.add_qset(iq) for iq in iqs]
                for v in vs:
                    self.add_validator(v)
                n = "_q" + str(self.qset_count)
                self.qset_count += 1
                self.qsets[n] = fqs
                self.graph.add_node(n, threshold=int(t))
                for member in set(vs) | set(iqs_nodes):
                    self.graph.add_edge(n, member)
                return n
            case _:
                raise ValueError(f"Invalid qset: {qset}")

    def __str__(self):
        # number qset nodes from 1 to n:
        # qset_nodes = {n for n in self.graph.nodes() if not n in self.validators}
        # qset_index = {n:i for i,n in enumerate(qset_nodes, 1)}
        # logical_validators = {n for n in self.validators if 'logical' in self.graph.nodes[n]}
        # logical_validators_index = {n:i for i,n in enumerate(logical_validators, 1)}
        # def node_repr(n):
        #     if n in self.validators:
        #         return f"{n}"
                # if not 'logical' in self.graph.nodes[n]:
                #     return f"{n}"
                # else:
                #     return f"_l{logical_validators_index[n]}"
            # else:
            #     return f"_q{qset_index[n]}"
        res = {n : f"({t}, {set(self.graph.successors(n))})"
                for n,t in self.graph.nodes('threshold')}
        return pformat(res)

    def threshold(self, n: Any) -> int:
        """
        Returns the threshold of the given node.
        """
        if 'threshold' in self.graph.nodes[n]:
            return self.graph.nodes[n]['threshold']
        elif self.graph.out_degree(n) == 1:
            return 1
        elif self.graph.out_degree(n) == 0:
            return -1
        else:
            raise ValueError(f"Node {n} has no threshold attribute and out-degree > 1")
    
    def qset_node_of(self, n: Any) -> tuple[int, frozenset]:
        """
        Returns the qset node of the given validator node (i.e. its successor).
        """
        assert n in self.validators
        assert self.graph.out_degree(n) == 1
        return next(self.graph.successors(n))
    
    @staticmethod
    def from_json(data : list, from_stellarbeat = False) -> 'FBASGraph':
        """
        Create a FBASGraph from a list of validators in serialized stellarbeat.io format.
        """
        # first do some validation
        validators = []
        keys = set()
        for v in data:
            if not isinstance(v, dict):
                logging.debug("Ignoring non-dict entry: %s", v)
                continue
            if 'publicKey' not in v:
                logging.debug(
                    "Entry is missing publicKey, skipping: %s", v)
                continue
            if (from_stellarbeat and (
                    ('isValidator' not in v or not v['isValidator'])
                    or ('isValidating' not in v or not v['isValidating']))):
                logging.debug(
                    "Ignoring non-validating validator: %s (name: %s)", v['publicKey'], v.get('name'))
                continue
            if 'quorumSet' not in v or v['quorumSet'] is None:
                logging.debug("Skipping validator missing quorumSet: %s", v['publicKey'])
                continue
            if v['publicKey'] in keys:
                logging.debug(
                    "Ignoring duplicate validator: %s", v['publicKey'])
                continue
            keys.add(v['publicKey'])
            validators.append(v)
        # now create the graph:
        fbas = FBASGraph()
        for v in validators:
            fbas.update_validator(v['publicKey'], v['quorumSet'], v)
        fbas.check_integrity()
        return fbas
    
    def is_qset_sat(self, q: Tuple[int, frozenset], s: Collection) -> bool:
        """
        Returns True if and only if q's agreement requirements are satisfied by s.
        """
        assert set(s) <= self.validators
        if all(c in self.validators for c in self.graph.successors(q)):
            assert q not in self.validators
            assert 'threshold' in self.graph.nodes[q] # canary
            return self.threshold(q) <= sum(1 for c in self.graph.successors(q) if c in s)
        else:
            return self.threshold(q) <= sum(1 for c in self.graph.successors(q) if c not in self.validators and self.is_qset_sat(c , s))
    
    def is_sat(self, n: Any, s: Collection) -> bool:
        """
        Returns True if and only if n's agreement requirements are satisfied by s.
        """
        assert n in self.validators
        if self.threshold(n) <= 0:
            return True
        else:
            return self.is_qset_sat(self.qset_node_of(n), s)

    def qset_nodes(self, n: Any) -> frozenset:
        """
        If n is a qset node, returns the set of graph nodes that form the full qset below n and including n.
        Otherwise just return {n}.
        """
        assert n in self.graph.nodes
        if n in self.validators:
            return frozenset([n])
        else:
            return frozenset([n]) | frozenset.union(*[self.qset_nodes(c) for c in self.graph.successors(n)])
    
    def is_quorum(self, vs: Collection) -> bool:
        """
        Returns True if and only if s is a non-empty quorum.
        Not efficient.
        """
        if not vs:
            return False
        assert set(vs) <= self.validators
        assert any([self.threshold(v) >= 0 for v in vs]) # we have a qset for at least one validator
        return all(self.is_sat(v, vs) for v in vs)
    
    def find_disjoint_quorums(self) -> Optional[tuple[set, set]]:
        """
        Naive, brute-force search for disjoint quorums.
        Warning: use only for very small fbas graphs.
        """
        assert len(self.validators) < 10
        quorums = [q for q in powerset(list(self.validators)) if self.is_quorum(q)]
        return next(((q1, q2) for q1 in quorums for q2 in quorums if not (q1 & q2)), None)
    
    def blocks(self, s : Collection, n : Any) -> bool:
        """
        Returns True if and only if s blocks v.
        """
        if self.threshold(n) <= 0:
            return False
        return self.threshold(n) + sum(1 for c in self.graph.successors(n) if c in s) > self.graph.out_degree(n)
    
    def closure(self, vs: Collection) -> frozenset:
        """
        Returns the closure of the set of validators vs.
        """
        assert set(vs) <= self.validators
        closure = set(vs)
        while True:
            new = {n for n in self.graph.nodes() - closure if self.blocks(closure, n)}
            if not new:
                return frozenset([v for v in closure if v in self.validators])
            closure |= new

    def flatten_diamonds(self) -> None:
        """
        Identify all the "diamonds" in the graph and "flatten" them.
        This creates a new logical validator in place of the diamond, and a 'logical' attribute set to True.
        A diamond is formed by a qset node show children have no other parent, whose threshold is non-zero and strictly greater than half, and that has a unique grandchild.
        This operation mutates the FBAS in place.
        It preserves both quorum intersection and non-intersection.

        NOTE: this is complex and doesn't seem that useful.
        """

        # a counter to create fresh logical validators:
        count = 1

        def collapse_diamond(n: Any) -> bool:
            """collapse diamonds with > 1/2 threshold"""
            nonlocal count
            assert n in self.graph.nodes
            if not all(n in self.validators for n in self.graph.successors(n)):
                return False
            # condition on threshold:
            if self.threshold(n) <= 1 or 2*self.threshold(n) < self.graph.out_degree(n)+1:
                return False
            # n must be its children's only parent:
            children = set(self.graph.successors(n))
            if not all(set(self.graph.predecessors(c)) == {n} for c in children):
                return False
            # n must have a unique grandchild:
            grandchildren = set.union(*[set(self.graph.successors(c)) for c in children])
            if len(grandchildren) != 1:
                return False
            # now collpase the diamond:
            grandchild = next(iter(grandchildren))
            logging.debug("Collapsing diamond at: %s", n)
            assert n not in self.validators # canary
            # first remove the node:
            parents = list(self.graph.predecessors(n))
            in_edges = [(p, n) for p in parents]
            self.graph.remove_node(n)
            # now add the new node:
            new_node = f"_l{count}"
            count += 1
            if n != grandchild:
                self.graph.add_edge(new_node, grandchild)
                self.update_validator(new_node, attrs={'threshold': 1, 'logical': True})
            else:
                self.update_validator(new_node, attrs={'threshold': 0, 'logical': True})
            # if some parents are validators, then we need to add a qset node:
            if any(p in self.validators for p in parents):
                new_qset = self.add_qset({'threshold': 1, 'validators': [new_node], 'innerQuorumSets': []})
                for e in in_edges:
                    self.graph.add_edge(e[0], new_qset)
            else:
                for e in in_edges:
                    self.graph.add_edge(e[0], new_node)
            return True

        # now collapse nodes until nothing changes:
        while True:
            for n in self.graph.nodes():
                if collapse_diamond(n):
                    self.check_integrity() # canary 
                    break
            else:
                return
            