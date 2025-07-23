"""
Federated byzantine agreement systems (FBAS) represented as directed graphs.
Each vertex in the graph is either a validator vertex or a quorum set vertex,
which has a threshold attribute.
"""

from copy import copy
from typing import Any, Literal, Optional, Dict, Tuple
from collections.abc import Collection
from itertools import chain, combinations
import logging
from pprint import pformat
import uuid
import networkx as nx
from networkx.classes.reportviews import NodeView
from python_fbas.utils import powerset
from python_fbas.config import get as get_config


class FBASGraph:
    """
    An FBAS graph is a directed graph. Each vertex is either a validator or a
    qset vertex, which may have a threshold attribute. Vertices are identified
    by strings.

    A validator vertex must have at most one sucessor in the graph, which must
    be a qset vertex, and has no threshold. If it does not have a successor,
    this means its quorum set is unknown. It may have attributes (e.g.
    homeDomain). When serializing/deserializing, the validator vertex id
    corresponds to its public key.

    A qset vertex may have any number of successors (including none), which may
    be validator or qset vertices. It must have a threshold between 0 and its
    number of successors.

    Two important invariants are that the subgraph of the qset vertices must be
    acyclic and that no two qset vertices may have the same threshold and same
    outgoing edges.

    TODO: rethink the interface; we need to maintain the invariants, so it would
    be better if clients did not have to access the graph directly.
    """

    _graph: nx.DiGraph  # TODO made private, don't use elsewhere
    # tracts which vertices in the graph are validator vertices:
    _validators: set[str]  # TODO should export only read-only view
    _qsets: Dict[Tuple[int, frozenset[str]], str]  # maps (threshold, members) to qset vertex ID

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._validators = set()
        self._qsets = dict()  # cache for qsets

    def __copy__(self) -> 'FBASGraph':
        fbas = FBASGraph()
        fbas._graph = nx.DiGraph(self._graph)
        fbas._validators = self._validators.copy()
        fbas._qsets = self._qsets.copy()
        return fbas

    def vertices(self, data: bool = False) -> NodeView:  # TODO: looks like it's never used with data=True
        return self._graph.nodes(data=data)

    def vertice_attrs(self, n: str) -> Dict[str, Any]:
        # TODO should we return a copy?
        return self._graph.nodes[n]

    def check_integrity(self) -> None:
        """Basic integrity checks; raises ValueError"""

        qset_vertices = set(self._qsets.values())

        # _validators and _qset must be disjoint:
        if self._validators & qset_vertices:
            raise ValueError(
                "Validators and qset vertices must be disjoint, but "
                f"{self._validators & qset_vertices} are in both")

        # the set of vertices in the graph is the union of validators and qset
        # vertices:
        union = self._validators | qset_vertices
        if self._graph.nodes() != union:
            raise ValueError(
                "Integrity check failed: the set of vertices in the graph is "
                "not equal to the union of _validators and _qsets.")

        # validators must have a qset successor or no successors at all:
        for v in self._validators:
            if self._graph.out_degree(v) > 1:
                raise ValueError(
                    f"Integrity check failed: validator {v} has more than one "
                    "successor, which is not allowed")
            if self._graph.out_degree(v) == 1:
                succ = next(self._graph.successors(v))
                if succ not in qset_vertices:
                    raise ValueError(
                        f"Integrity check failed: validator {v} has a successor "
                        f"{succ} that is not a qset vertex")

        # a qset vertice must have a threshold attribute between 0 and its
        # out-degree:
        for q in qset_vertices:
            threshold = self._graph.nodes[q].get('threshold', None)
            if threshold and (threshold < 0
                              or threshold > self._graph.out_degree(q)):
                raise ValueError(
                    f"Integrity check failed: threshold of {q} not in "
                    f"[0, out_degree={self._graph.out_degree(q)}]")

        # The qset subgraph must be loop-free:
        qset_vertices = [q for q in self._graph.nodes()
                         if q not in self._validators]
        qset_only_graph = self._graph.subgraph(qset_vertices)
        if not nx.is_directed_acyclic_graph(qset_only_graph):
            raise ValueError(
                "Integrity check failed: the qset subgraph has a cycle: "
                f"{nx.find_cycle(qset_only_graph)}")

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

    def add_validator(self, v: str, qset: Optional[str] = None, **attrs) -> None:
        """Add a validator to the graph."""
        if v in self._validators:
            raise ValueError(f"Validator {v} already exists in the graph")
        self.update_validator(v, qset, **attrs)

    def update_validator(self, v: str, qset: Optional[str] = None, **attrs) -> None:
        self._graph.add_node(v, **attrs)
        self._validators.add(v)
        if qset:
            # first remove all existing edges from v:
            if v in self._validators:
                out_edges = list(self._graph.out_edges(v))
                self._graph.remove_edges_from(out_edges)
            # then add the new qset:
            self._graph.add_edge(v, qset)

    def add_qset(self, threshold: int, components: Collection[str],
                 qset_id: Optional[str] = None) -> str:
        """
        Add a quorum set vertex to the graph. If a qset with the same threshold
        and components already exists, returns the existing qset's ID instead of
        creating a duplicate.

        Args:
            threshold: The threshold for the quorum set
            members: List of vertex IDs (validators or other qset vertices) that are members. Those must be in the graph.
            qset_id: Optional ID for the qset vertex. If not provided, a UUID-based ID will be generated. If provided but a duplicate qset exists, a warning is emitted.

        Returns:
            The ID of the qset vertex (either existing or newly created)
        """
        # Validate inputs
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        if threshold > len(components):
            raise ValueError(f"Threshold {threshold} cannot exceed {len(components)} (the number of components of the qset)")

        # first check if a qset with the same threshold and components already exists:
        components_set = frozenset(components)
        qset_spec = (threshold, components_set)
        if qset_spec in self._qsets:
            existing_qset_id = self._qsets[qset_spec]
            # warn only if qset_id was provided and does not match the existing one:
            if qset_id is not None and qset_id != existing_qset_id:
                logging.warning(
                    f"QSet with threshold {threshold} and components {components_set} already exists with ID {existing_qset_id}, "
                    f"but a different ID {qset_id} was provided. Using existing ID instead.")
            return existing_qset_id

        # Check that all components are in the graph:
        for member in components:
            if member not in self._graph:
                raise ValueError(f"Component {member} is not in the graph")

        # Create qset ID if not provided
        if qset_id is None:
            qset_id = "_q" + uuid.uuid4().hex
        else:
            # Validate that the ID doesn't already exist
            if qset_id in self._graph:
                raise ValueError(f"Vertex with ID {qset_id} already exists in the graph")
            # Validate that it's not in the validators set
            if qset_id in self._validators:
                raise ValueError(f"ID {qset_id} is already used by a validator")

        # Create the qset vertex
        self._graph.add_node(qset_id, threshold=threshold)
        self._qsets[qset_spec] = qset_id

        # Add edges to all members
        for member in components:
            self._graph.add_edge(qset_id, member)

        return qset_id

    def __str__(self) -> str:
        def info(v: str) -> str:
            if self._graph.out_degree(v) > 0:
                return f"({self.threshold(v)}, {set(self._graph.successors(v))})"
            else:
                return "unknown"
        res = {v: info(v) for v in self.vertices()}
        return pformat(res)

    def threshold(self, n: str) -> Optional[int]:
        """
        Returns the threshold of the given vertex.
        """
        return self.vertice_attrs(n)['threshold']

    def qset_vertex_of(self, n: str) -> Optional[str]:
        assert n in self._validators
        return next(self._graph.successors(n), None)

    def is_qset_sat(self, q: str, s: Collection[str]) -> bool:
        """
        Returns True if and only if q's agreement requirements are satisfied by s.
        NOTE: this is a recursive function and it could blow the stack if the
        qset graph is very deep.
        """
        assert set(s) <= self._validators
        if all(c in self._validators for c in self._graph.successors(q)):
            assert q not in self._validators
            assert 'threshold' in self.vertice_attrs(q)  # canary
            return self.threshold(q) <= sum(
                1 for c in self._graph.successors(q) if c in s)
        else:
            return self.threshold(q) <= \
                sum(1 for c in self._graph.successors(q) if
                    c not in self._validators and self.is_qset_sat(c, s)
                    or c in self._validators and c in s)

    def is_sat(
            self,
            n: str,
            s: Collection[str],
            over_approximate: bool = True) -> bool:
        """
        Returns True if and only if n's agreement requirements are satisfied by s.
        """
        assert n in self._validators
        if self._graph.out_degree(n) == 0:
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
        assert set(vs) <= self._validators
        return all(self.is_sat(v, vs, over_approximate) for v in vs)

    def find_disjoint_quorums(self) -> Optional[tuple[set[str], set[str]]]:
        """
        Naive, brute-force search for disjoint quorums.
        Warning: use only for very small fbas graphs.
        """
        assert len(self._validators) < 10
        quorums = [
            q for q in powerset(
                list(
                    self._validators)) if self.is_quorum(
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
        remaining_successors = len(set(self._graph.successors(n)) - set(s))
        return remaining_successors < self.threshold(n)

    def closure(self, vs: Collection[str]) -> frozenset[str]:
        """
        Returns the closure of the set of validators vs.
        """
        assert set(vs) <= self._validators
        closure = set(vs)
        while True:
            new = {
                n for n in self.vertices() -
                closure if self.blocks(
                    closure,
                    n)}
            if not new:
                return frozenset([v for v in closure if v in self._validators])
            closure |= new

    def project(self, vs: Collection[str]) -> 'FBASGraph':
        """
        Returns a new fbas that only contains the validators reachable from the set vs.
        """
        assert set(vs) <= self._validators
        reachable = set.union(
            *[set(nx.descendants(self._graph, v)) | {v} for v in vs])
        fbas = copy(self)
        fbas._graph = nx.subgraph(self._graph, reachable)
        fbas._validators = reachable & self._validators
        return fbas

    def groups_dict(self, group_by: str = 'homeDomain') -> dict[str, set[str]]:
        """
        Computes a dict mapping group names to sets of member validators.
        Groups are determined by the `group_by` attribute of validator nodes.
        Validators without a value for the `group_by` attribute are skipped.
        """
        groups: dict[str, set[str]] = {}
        for v in self._validators:
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
        assert n in self._graph
        if n in self._validators:
            return True
        return all(c in self._validators for c in self._graph.successors(n)) \
            and 2 * self.threshold(n) > self._graph.out_degree(n)

    def intersection_bound_heuristic(self, n1: str, n2: str) -> int:
        """
        If n1 and n2's children are self-intersecting,
        then return the mininum number of children in common in two sets that satisfy n1 and n2.
        """
        assert n1 in self._graph and n2 in self._graph
        assert n1 not in self._validators and n2 not in self._validators
        if all(
            self.self_intersecting(c) for c in chain(
                self._graph.successors(n1),
                self._graph.successors(n2))):
            o1, o2 = self._graph.out_degree(n1), self._graph.out_degree(n2)
            t1, t2 = self.threshold(n1), self.threshold(n2)
            common_children = set(
                self._graph.successors(n1)) & set(
                self._graph.successors(n2))
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
        mscc = max(nx.strongly_connected_components(self._graph), key=len)
        validators_with_qset = {
            v for v in self._validators if self._graph.out_degree(v) == 1}
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
                self._validators,
                2) if v1 != v2)
