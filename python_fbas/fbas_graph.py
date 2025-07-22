"""
Federated byzantine agreement systems (FBAS) represented as directed graphs.
Each vertex in the graph is either a validator vertex or a quorum set vertex,
which has a threshold attribute.
"""

from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict
from collections.abc import Collection, Set
from itertools import chain, combinations
import logging
from pprint import pformat
import uuid
import networkx as nx
from networkx.classes.reportviews import NodeView
from python_fbas.utils import powerset
from python_fbas.config import get as get_config


@dataclass(frozen=True)
class QSet:
    """
    Represents a stellar-core quorum set in a unique, hashable way. Note that a
    quorum set is _not_ a set of quorums.  Instead, a quorum set represents
    agreement requirements. For quorums, see `is_quorum` in `FBASGraph`.
    """
    threshold: int
    validators: Set[str]
    inner_quorum_sets: Set['QSet']


class FBASGraph:
    """
    An FBAS graph is a directed graph. Each vertex is either a validator or a
    qset vertex, which may have a threshold attribute. Vertices are identified
    by strings.

    A validator vertex must have at most one sucessor in the graph, which must
    be a qset vertex, and has no threshold. If it does not have a successor,
    this means its quorum set is unknown.

    A qset vertex may have any number of successors (including none), which may
    be validator or qset vertices, but which must not include itself. It must
    have a threshold between 0 and its number of successors. The subgraph of the
    qset vertices must be acyclic.
    """

    graph: nx.DiGraph
    # tracts which vertices in the graph are validator vertices:
    validators: set[str]

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.validators = set()

    def __copy__(self) -> 'FBASGraph':
        fbas = FBASGraph()
        fbas.graph = nx.DiGraph(self.graph)
        fbas.validators = self.validators.copy()
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
                        f"Integrity check failed: threshold of {n} not in "
                        f"[0, out_degree={self.graph.out_degree(n)}]")
            if n in self.validators:
                # threshold is not explicitly set for validators, so it should
                # not appear in the attributes of n:
                assert 'threshold' not in attrs
                # a validator either has one successor (its qset vertex) or no
                # successors (in case we do not know its agreement
                # requirements):
                if self.graph.out_degree(n) > 1:
                    raise ValueError(
                        f"Integrity check failed: validator {n} has an "
                        "out-degree greater than 1 "
                        f"({self.graph.out_degree(n)})")
                # a validator's successor must be a qset vertex:
                if self.graph.out_degree(n) == 1:
                    assert next(
                        self.graph.successors(n)) not in self.validators
            if n in self.graph.successors(n):
                raise ValueError(
                    f"Integrity check failed: vertex {n} has a self-loop")

        # Check that the qset subgraph is loop-free
        qset_vertices = [q for q in self.graph.nodes()
                         if q not in self.validators]
        qset_only_graph = self.graph.subgraph(qset_vertices)
        if not nx.is_directed_acyclic_graph(qset_only_graph):
            raise ValueError(
                "Integrity check failed: the qset subgraph has a cycle")

        # Check for duplicate qsets (same threshold and same successors)
        qset_signatures = {}
        for q in qset_vertices:
            threshold = self.threshold(q)
            successors = tuple(sorted(self.graph.successors(q)))
            signature = (threshold, successors)

            if signature in qset_signatures:
                raise ValueError(
                    f"Integrity check failed: duplicate qsets found with same threshold {threshold} "
                    f"and members {list(successors)}: {qset_signatures[signature]} and {q}")
            qset_signatures[signature] = q

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

    def add_validator(self, v: str) -> None:
        """Add a validator to the graph."""
        self.graph.add_node(v)
        self.validators.add(v)

    def update_validator(self, v: str, qset: str) -> None:
        # first remove all existing edges from v:
        if v in self.validators:
            out_edges = list(self.graph.out_edges(v))
            self.graph.remove_edges_from(out_edges)
        # then add the new qset:
        self.graph.add_edge(v, qset)

    def add_qset(self, threshold: int, members: list[str], qset_id: Optional[str] = None) -> str:
        """
        Add a quorum set vertex to the graph. If a qset with the same threshold
        and members already exists, returns the existing qset's ID instead of creating a duplicate.

        Args:
            threshold: The threshold for the quorum set
            members: List of vertex IDs (validators or other qset vertices) that are members
            qset_id: Optional ID for the qset vertex. If not provided, a UUID-based ID will be generated.
                    If provided but a duplicate qset exists, a warning is emitted.

        Returns:
            The ID of the qset vertex (either existing or newly created)
        """
        # Validate inputs
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        if threshold > len(members):
            raise ValueError(f"Threshold {threshold} cannot exceed number of members {len(members)}")

        # Validate that all members exist in the graph
        for member in members:
            if member not in self.graph:
                raise ValueError(f"Member {member} does not exist in the graph")

        # Check for existing qset with same threshold and members
        members_set = set(members)
        qset_nodes = [q for q in self.graph.nodes() if q not in self.validators]

        for existing_qset_id in qset_nodes:
            if (self.threshold(existing_qset_id) == threshold and
                set(self.graph.successors(existing_qset_id)) == members_set):

                # Found a duplicate - warn if qset_id was provided
                if qset_id is not None:
                    logging.warning(
                        "Qset with threshold %d and members %s already exists as %s. "
                        "Returning existing qset instead of creating new one with ID %s.",
                        threshold, sorted(members), existing_qset_id, qset_id)

                return existing_qset_id

        # No duplicate found, create new qset
        # Create qset ID if not provided
        if qset_id is None:
            qset_id = "_q" + uuid.uuid4().hex
        else:
            # Validate that the ID doesn't already exist
            if qset_id in self.graph:
                raise ValueError(f"Vertex with ID {qset_id} already exists in the graph")
            # Validate that it's not in the validators set
            if qset_id in self.validators:
                raise ValueError(f"ID {qset_id} is already used by a validator")

        # Create the qset vertex
        self.graph.add_node(qset_id, threshold=threshold)

        # Add edges to all members
        for member in members:
            self.graph.add_edge(qset_id, member)

        return qset_id


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

    def is_sat(
            self,
            n: str,
            s: Collection[str],
            over_approximate: bool = True) -> bool:
        """
        Returns True if and only if n's agreement requirements are satisfied by s.
        """
        assert n in self.validators
        if self.graph.out_degree(n) == 0:
            return over_approximate
        return self.is_qset_sat(self.qset_vertex_of(n), s)

    def is_quorum(
            self,
            vs: Collection[str],
            *,
            over_approximate: bool = True) -> bool:
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

