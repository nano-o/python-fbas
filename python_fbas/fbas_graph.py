"""
Federated Byzantine Agreement System (FBAS) represented as directed graphs.

TODO: Think about how to better deal with validators that do not have a qset.
"""

from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict
from collections.abc import Collection, Set
from itertools import chain, combinations
import logging
from pprint import pformat
import networkx as nx
from networkx.classes.reportviews import NodeView
from python_fbas.utils import powerset
from python_fbas.config import get as get_config


@dataclass(frozen=True)
class QSet:
    """
    Represents a quorum set. Note that a quorum set is _not_ a set of quorums.
    Instead, a quorum set represents agreement requirements. For quorums, see
    `is_quorum` in `FBASGraph`.
    """
    threshold: int
    validators: Set[str]
    inner_quorum_sets: Set['QSet']

    @staticmethod
    def make(qset: Dict[str, Any]) -> 'QSet':
        """
        Expects a JSON-serializable quorum-set (in stellarbeat.io format) and
        returns a QSet instance.
        """
        match qset:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
                threshold = int(t)
                validators = frozenset(vs)
                inner_qsets = frozenset(QSet.make(iq) for iq in iqs)
                card = len(validators) + len(inner_qsets)
                if not (0 <= threshold <= card):
                    logging.info(
                        "QSet validation failed: threshold=%d not in range [0, %d] for qset with %d validators and %d inner quorum sets. Qset: %s",
                        threshold, card, len(validators), len(inner_qsets), qset)
                    raise ValueError(f"Invalid qset threshold {threshold} (must be 0 <= threshold <= {card}): {qset}")
                return QSet(threshold, validators, inner_qsets)
            case _:
                logging.info(
                    "QSet.make failed: qset does not match expected format. Expected keys: threshold, validators, innerQuorumSets. Actual keys: %s. Qset: %s",
                    list(qset.keys()) if isinstance(qset, dict) else "not a dict", qset)
                raise ValueError(f"Invalid qset format (expected dict with threshold, validators, innerQuorumSets): {qset}")


class FBASGraph:
    """
    An FBAS graph is a directed graph. Each vertex is either a validator or a
    qset vertex, and may have a threshold attribute. Vertices are identified by
    strings.

    A validator vertex must have at most one sucessor in the graph, which must
    be a qset vertex, and has no threshold. If it does not have a successor,
    this means its quorum set is unknown.

    A qset vertex may have any number of successors (including none), which may
    be validator or qset vertices, but which must not include itself. It must
    have a threshold between 0 and its number of successors.
    """
    graph: nx.DiGraph
    # only a subset of the vertices in the graph represent validators:
    validators: set[str]
    qset_count = 1  # used to create unique qset identifiers
    # maps qset vertices (str) to their associated qset:
    qsets: dict[str, QSet]  # TODO: how to keep in sync with the graph?

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.validators = set()
        self.qsets = dict()

    def __copy__(self) -> 'FBASGraph':
        fbas = FBASGraph()
        fbas.graph = nx.DiGraph(self.graph)
        fbas.validators = self.validators.copy()
        fbas.qset_count = self.qset_count
        fbas.qsets = self.qsets.copy()
        return fbas

    def vertices(self, data: bool = False) -> NodeView:
        return self.graph.nodes(data=data)

    def vertice_attrs(self, n: str) -> Dict[str, Any]:
        return self.graph.nodes[n]

    def check_integrity(self) -> None:
        """Basic integrity checks"""
        # check that all validators are in the graph:
        if not self.validators <= self.vertices():
            missing = self.validators - self.vertices()
            raise ValueError(
                f"Some validators are not in the graph: {missing}")
        for n, attrs in self.vertices(data=True):
            if 'threshold' not in attrs:
                assert n in self.validators
            else:
                if attrs['threshold'] < 0 \
                   or attrs['threshold'] > self.graph.out_degree(n):
                    raise ValueError(
                        f"Integrity check failed: threshold of {n} not in [0, out_degree={self.graph.out_degree(n)}]")
            if n in self.validators:
                # threshold is not explicitly set for validators, so it should
                # not appear in the attributes of n:
                assert 'threshold' not in attrs
                # a validator either has one successor (its qset vertex) or no
                # successors (in case we do not know its agreement
                # requirements):
                if self.graph.out_degree(n) > 1:
                    raise ValueError(
                        f"Integrity check failed: validator {n} has an out-degree greater than 1 ({self.graph.out_degree(n)})")
                # a validator's successor must be a qset vertex:
                if self.graph.out_degree(n) == 1:
                    assert next(
                        self.graph.successors(n)) not in self.validators
            else:
                assert n in self.qsets.keys()
                assert self.qsets[n] == self.compute_qset(n)
            if n in self.graph.successors(n):
                raise ValueError(
                    f"Integrity check failed: vertex {n} has a self-loop")

    def format_validator(self, validator_id: str) -> str:
        """
        Formats a validator's string representation based on the
        `validator_display` configuration setting.
        """
        cfg = get_config()
        attrs = self.vertice_attrs(validator_id)
        name = attrs.get('name')

        if cfg.validator_display == 'id':
            return validator_id
        if cfg.validator_display == 'name':
            return name if name else validator_id
        # 'both' is the default
        if name:
            return f"{validator_id} ({name})"
        return validator_id

    def add_validator(self, v: Any) -> None:
        """Add a validator to the graph."""
        self.graph.add_node(v)
        self.validators.add(v)

    def update_validator(self, v: Any, qset: Optional[Dict[str, Any]] = None,
                         attrs: Optional[Dict[str, Any]] = None) -> None:
        """
        Add the validator v to the graph if it does not exist, using the
        supplied qset and attributes. Otherwise:
            - Update its attributes with attrs (existing attributes not in attrs
                remain unchanged).
            - Replace its outgoing edge with an edge to the given qset.
        Expects a qset, if given, in JSON-serializable stellarbeat.io format.
        """
        if attrs:
            # check that 'threshold' is not in attrs, as it's a reserved
            # attribute
            if 'threshold' in attrs:
                raise ValueError(
                    "'threshold' is reserved and cannot be passed as an attribute")
            self.graph.add_node(v, **attrs)
        else:
            self.graph.add_node(v)
        self.validators.add(v)
        if qset:
            try:
                fqs = self.add_qset(qset)
            except ValueError as e:
                logging.info(
                    "Failed to add qset for validator %s: %s. Qset data: %s", v, e, qset)
                return
            out_edges = list(self.graph.out_edges(v))
            self.graph.remove_edges_from(out_edges)
            self.graph.add_edge(v, fqs)

    def add_qset(self, qset: Dict[str, Any]) -> str:
        """
        Takes a qset as a JSON-serializable dict in stellarbeat.io format.
        Returns the qset if it already exists, otherwise adds it to the graph.
        """
        match qset:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
                fqs = QSet.make(qset)
                if fqs in self.qsets.values():
                    return next(k for k, v in self.qsets.items() if v == fqs)
                iqs_vertices = [self.add_qset(iq) for iq in iqs]
                for v in vs:
                    self.add_validator(v)
                n = "_q" + str(self.qset_count)
                self.qset_count += 1
                self.qsets[n] = fqs
                self.graph.add_node(n, threshold=int(t))
                for member in set(vs) | set(iqs_vertices):
                    self.graph.add_edge(n, member)
                return n
            case _:
                logging.info(
                    "add_qset failed: qset does not match expected format. Expected keys: threshold, validators, innerQuorumSets. Actual keys: %s. Qset: %s",
                    list(qset.keys()) if isinstance(qset, dict) else "not a dict", qset)
                raise ValueError(f"Invalid qset format (expected dict with threshold, validators, innerQuorumSets): {qset}")

    def __str__(self) -> str:
        def info(v: str) -> str:
            if self.graph.out_degree(v) > 0:
                return f"({self.threshold(v)}, {set(self.graph.successors(v))})"
            else:
                return "no qset information"
        res = {v: info(v) for v in self.vertices()}
        return pformat(res)

    def threshold(self, n: Any) -> int:
        """
        Returns the threshold of the given vertex.
        """
        if n in self.validators:
            return 1 if self.graph.out_degree(n) == 1 else 0
        if 'threshold' in self.vertice_attrs(n):
            return self.vertice_attrs(n)['threshold']
        raise ValueError(f"QSet vertex {n} has no threshold attribute")

    def qset_vertex_of(self, n: str) -> str:
        """
        n must be a validator vertex that has a successor.
        Returns the successor of n, which is supposed to be a qset vertex.
        """
        assert n in self.validators
        assert self.graph.out_degree(n) == 1
        return next(self.graph.successors(n))

    def compute_qset(self, qset_vertex: str) -> QSet:
        """
        Recursively computes the QSet associated with the given qset vertex.
        """
        assert qset_vertex not in self.validators
        threshold = self.threshold(qset_vertex)
        # validators are the children of the qset vertex that are validators:
        validators = frozenset(v for v in self.graph.successors(
            qset_vertex) if v in self.validators)
        # inner_qsets are the children of the qset vertex that are qset
        # vertices:
        inner_qsets = frozenset(self.compute_qset(q) for q in self.graph.successors(
            qset_vertex) if q not in self.validators)
        return QSet(threshold, validators, inner_qsets)

    def qset_of(self, n: str) -> Optional[QSet]:
        """
        Computes the QSet associated with the given vertex n based on the graph (does not use the qsets dict).
        n must be a validator vertex.
        """
        assert n in self.validators
        # if n has no successors, then we don't know its qset:
        if self.graph.out_degree(n) == 0:
            return None
        return self.compute_qset(self.qset_vertex_of(n))

    @staticmethod
    def from_json(data: list[Dict[str, Any]], from_stellarbeat: bool = False) -> 'FBASGraph':
        """
        Create a FBASGraph from a list of validators in serialized
        stellarbeat.io format.
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
                    "Ignoring non-validating validator: %s (name: %s)",
                    v['publicKey'],
                    v.get('name'))
                continue
            if 'quorumSet' not in v or v['quorumSet'] is None:
                logging.debug(
                    "Skipping validator missing quorumSet: %s",
                    v['publicKey'])
                continue
            if v['publicKey'] in keys:
                logging.warning(
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

    def is_qset_sat(self, q: str, s: Collection[str]) -> bool:
        """
        Returns True if and only if q's agreement requirements are satisfied by s.
        NOTE this is a recursive function and it could blow the stack if the qset graph is very deep.
        """
        assert set(s) <= self.validators
        if all(c in self.validators for c in self.graph.successors(q)):
            assert q not in self.validators
            assert 'threshold' in self.vertice_attrs(q)  # canary
            return self.threshold(q) <= sum(
                1 for c in self.graph.successors(q) if c in s)
        else:
            return self.threshold(q) <= \
                sum(1 for c in self.graph.successors(q) if
                    c not in self.validators and self.is_qset_sat(c, s)
                    or c in self.validators and c in s)

    def is_sat(self, n: str, s: Collection[str], over_approximate: bool = True) -> bool:
        """
        Returns True if and only if n's agreement requirements are satisfied by s.
        """
        assert n in self.validators
        if self.graph.out_degree(n) == 0:
            return over_approximate
        return self.is_qset_sat(self.qset_vertex_of(n), s)

    def is_quorum(
            self, vs: Collection[str], *, over_approximate: bool = True) -> bool:
        """
        Returns True if and only if s is a non-empty quorum. If
        `over_approximate` is true, we consider that a validator that has no
        qset has no requirement; otherwise, if vs contains a validator that has
        no qset, then it is not a quorum.
        """
        if not vs:
            return False
        assert set(vs) <= self.validators
        return all(self.is_sat(v, vs, over_approximate) for v in vs)

    def find_disjoint_quorums(self) -> Optional[tuple[set[str], set[str]]]:
        """
        Naive, brute-force search for disjoint quorums.
        Warning: use only for very small fbas graphs.
        """
        assert len(self.validators) < 10
        quorums = [
            q for q in powerset(
                list(
                    self.validators)) if self.is_quorum(
                q,
                over_approximate=True)]
        return next(((q1, q2)
                    for q1 in quorums for q2 in quorums if not (q1 & q2)), None)

    def blocks(self, s: Collection[str], n: Any) -> bool:
        """
        Returns True if and only if s blocks v.
        TODO: should there be an `overapproximate` parameter?
        """
        if self.threshold(n) == 0:
            return False
        remaining_successors = len(set(self.graph.successors(n)) - set(s))
        return remaining_successors < self.threshold(n)

    def closure(self, vs: Collection[str]) -> frozenset[str]:
        """
        Returns the closure of the set of validators vs.
        """
        assert set(vs) <= self.validators
        closure = set(vs)
        while True:
            new = {
                n for n in self.vertices() -
                closure if self.blocks(
                    closure,
                    n)}
            if not new:
                return frozenset([v for v in closure if v in self.validators])
            closure |= new

    def project(self, vs: Collection[str]) -> 'FBASGraph':
        """
        Returns a new fbas that only contains the validators reachable from the set vs.
        """
        assert set(vs) <= self.validators
        reachable = set.union(
            *[set(nx.descendants(self.graph, v)) | {v} for v in vs])
        fbas = copy(self)
        fbas.graph = nx.subgraph(self.graph, reachable)
        fbas.validators = reachable & self.validators
        fbas.qsets = {k: v for k, v in self.qsets.items() if k in reachable}
        return fbas

    def groups_dict(self, group_by: str = 'homeDomain') -> dict[str, set[str]]:
        """
        Computes a dict mapping group names to sets of member validators.
        Groups are determined by the `group_by` attribute of validator nodes.
        Validators without a value for the `group_by` attribute are skipped.
        """
        groups: dict[str, set[str]] = {}
        for v in self.validators:
            attrs = self.vertice_attrs(v)
            if group_by in attrs and attrs[group_by]:
                group_name = attrs[group_by]
                groups.setdefault(group_name, set()).add(v)
        return groups

    def restrict_to_reachable(self, v: str) -> 'FBASGraph':
        """
        Returns a new fbas that only contains what's reachable from v.
        """
        return self.project({v})

    # Fast heuristic checks:

    def self_intersecting(self, n: str) -> bool:
        """
        Whether n is self-intersecting
        """
        assert n in self.graph
        if n in self.validators:
            return True
        return all(c in self.validators for c in self.graph.successors(n)) \
            and 2 * self.threshold(n) > self.graph.out_degree(n)

    def intersection_bound_heuristic(self, n1: str, n2: str) -> int:
        """
        If n1 and n2's children are self-intersecting,
        then return the mininum number of children in common in two sets that satisfy n1 and n2.
        """
        assert n1 in self.graph and n2 in self.graph
        assert n1 not in self.validators and n2 not in self.validators
        if all(
            self.self_intersecting(c) for c in chain(
                self.graph.successors(n1),
                self.graph.successors(n2))):
            o1, o2 = self.graph.out_degree(n1), self.graph.out_degree(n2)
            t1, t2 = self.threshold(n1), self.threshold(n2)
            common_children = set(
                self.graph.successors(n1)) & set(
                self.graph.successors(n2))
            c = len(common_children)
            # worst-case number of common children among t1 children of n1
            m1 = (t1 + c) - o1
            # worst-case number of common children among t2 children of n2
            m2 = (t2 + c) - o2
            # return worst-case overlap if we pick m1 among c and m2 among c:
            return max((m1 + m2) - c, 0)
        else:
            return 0

    def fast_intersection_check(self) -> Literal['true', 'unknown']:
        """
        This is a fast, sound, but incomplete heuristic to check whether all of a FBAS's quorums intersect (i.e. is intertwined). It does not rely on a SAT solver.
        It may return 'unknown' even if the FBAS is intertwined (i.e. a false negative), but, if it returns 'true', then the FBAS is guaranteed intertwined (there are no false positives).
        NOTE: ignores validators for which we don't have a qset (because we don't know what their quorums might be).

        We use an important properties of FBASs: if a set of validators S is intertwined, then the closure of S is also intertwined.
        Thus our strategy is to try to find a small intertwined set whose closure covers all validators.

        To find such a set, we look inside the maximal strongly-connected component (the mscc) of the FBAS graph. First, we build a new graph over the validators in the mscc where there is an edge between v1 and v2 when v1 and v2 are intertwined, as computed by a sound but incomplete heuristic. Since the heuristic is sound, we know that any clique in this new graph is an intertwined set.
        """
        # first obtain a max scc:
        mscc = max(nx.strongly_connected_components(self.graph), key=len)
        validators_with_qset = {
            v for v in self.validators if self.graph.out_degree(v) == 1}
        mscc_validators = mscc & validators_with_qset
        # then create a graph over the validators in mscc where there is an
        # edge between v1 and v2 iff their qsets have a non-zero intersection
        # bound
        g = nx.Graph()
        for v1, v2 in combinations(mscc_validators, 2):
            if v1 != v2:
                q1 = self.qset_vertex_of(v1)
                q2 = self.qset_vertex_of(v2)
                if self.intersection_bound_heuristic(q1, q2) > 0:
                    g.add_edge(v1, v2)
                else:
                    logging.debug(
                        "Non-intertwined max-scc validators: %s and %s", v1, v2)
        # next, we try to find a clique such that the closure of the clique
        # contains all validators:
        max_tries = 100
        cliques = nx.find_cliques(g)  # I think this is a generator
        for _ in range(1, max_tries + 1):
            try:
                clique = next(cliques)
            except StopIteration:
                logging.debug(
                    "No clique whose closure covers the validators found")
                return 'unknown'
            if validators_with_qset <= self.closure(clique):
                return 'true'
            else:
                logging.debug(
                    "Validators not covered by clique: %s",
                    validators_with_qset -
                    self.closure(clique))
        return 'unknown'

    def splitting_set_bound(self) -> int:
        """
        Computes a lower bound on the mimimum splitting-set size. We just take the minimum of the intersection bound over all pairs of validators.
        """
        return min(
            self.intersection_bound_heuristic(
                self.qset_vertex_of(v1),
                self.qset_vertex_of(v2)) for v1,
            v2 in combinations(
                self.validators,
                2) if v1 != v2)

    def flatten_diamonds(self) -> None:
        """
        Identify all the "diamonds" in the graph and "flatten" them.  This
        creates a new logical validator in place of the diamond, and a 'logical'
        attribute set to True.  A diamond is formed by a qset vertex whose
        children have no other parent, whose threshold is non-zero and strictly
        greater than half, and that has a unique grandchild.  This operation
        mutates the FBAS in place.  It preserves both quorum intersection and
        non-intersection.

        NOTE: this is complex and doesn't seem that useful.
        """

        # a counter to create fresh logical validators:
        count = 1

        def collapse_diamond(n: Any) -> bool:
            """collapse diamonds with > 1/2 threshold"""
            nonlocal count
            assert n in self.vertices()
            if not all(n in self.validators for n in self.graph.successors(n)):
                return False
            # condition on threshold:
            if self.threshold(n) <= 1 or 2 * \
                    self.threshold(n) < self.graph.out_degree(n) + 1:
                return False
            # n must be its children's only parent:
            children = set(self.graph.successors(n))
            if not all(set(self.graph.predecessors(c))
                       == {n} for c in children):
                return False
            # n must have a unique grandchild:
            grandchildren = set.union(
                *[set(self.graph.successors(c)) for c in children])
            if len(grandchildren) != 1:
                return False
            # now collpase the diamond:
            grandchild = next(iter(grandchildren))
            logging.debug("Collapsing diamond at: %s", n)
            assert n not in self.validators  # canary
            # first remove the vertex:
            parents = list(self.graph.predecessors(n))
            in_edges = [(p, n) for p in parents]
            self.graph.remove_node(n)
            # now add the new vertex:
            new_vertex = f"_l{count}"
            count += 1
            self.update_validator(new_vertex, attrs={'logical': True})
            if n != grandchild:
                self.graph.add_edge(new_vertex, grandchild)
            else:
                empty = self.add_qset(
                    {'threshold': 0, 'validators': [], 'innerQuorumSets': []})
                self.graph.add_edge(new_vertex, empty)
            # TODO: can't we remove the children of n?
            # if some parents are validators, then we need to add a qset
            # vertex:
            if any(p in self.validators for p in parents):
                new_qset = self.add_qset({'threshold': 1, 'validators': [
                                         new_vertex], 'innerQuorumSets': []})
                for e in in_edges:
                    self.graph.add_edge(e[0], new_qset)
            else:
                for e in in_edges:
                    self.graph.add_edge(e[0], new_vertex)
            # fixup the qsets dict:
            # TODO: not efficient, we should only update what's changed
            for n in self.graph.nodes():
                if n not in self.qsets:
                    continue
                self.qsets[n] = self.compute_qset(n)
            return True

        # now collapse vertices until nothing changes:
        while True:
            for n in self.vertices():
                if collapse_diamond(n):
                    self.check_integrity()  # canary
                    break
            else:
                return
