"""
SAT-based analysis of FBAS graphs
"""

import logging
from typing import Any, Optional, Tuple, Collection, Callable, Set
from itertools import combinations
from functools import partial
import networkx as nx
from pysat.formula import CNF, WCNF
from python_fbas import solver as slv
from python_fbas.fbas_graph import FBASGraph
from python_fbas.propositional_logic import (
    And, Or, Implies, Atom, Formula, Card, Not, equiv, variables, to_cnf,
    atoms_of_clauses, decode_model
)
import python_fbas.config as config
from python_fbas.utils import timed
try:
    from pyqbf.formula import PCNF
    HAS_QBF = True
except ImportError:
    HAS_QBF = False


class Tagger:
    """
    Encapsulates a tag for creating and identifying tagged atoms.
    Uses a tuple `(tag, base_variable)` as the atom's identifier.
    """

    def __init__(self, tag: str):
        """Initializes the Tagger with a specific tag string."""
        self._tag = tag

    def atom(self, base_variable: Any) -> Atom:
        """Creates a new Atom with a tagged identifier."""
        # The identifier is a tuple for robustness, e.g., ('in_quorum', 'NodeA')
        return Atom((self._tag, base_variable))

    def is_tagged(self, identifier: Any) -> bool:
        """Checks if the given identifier is a tuple tagged by this Tagger."""
        return (
            isinstance(identifier, tuple) and
            len(identifier) == 2 and
            identifier[0] == self._tag
        )

    def strip_tag(self, identifier: Any) -> Any:
        """
        Extracts the base variable from a tagged identifier.
        Raises ValueError if the identifier is not correctly tagged.
        """
        if not self.is_tagged(identifier):
            raise ValueError(
                f"Identifier '{identifier}' is not tagged by '{self._tag}'")
        return identifier[1]


def extract_true_tagged_variables(model: list[int],
                                  tagger: Tagger) -> Set[Any]:
    """
    Given a pysat model (a list of ints), it decodes the model, filters for
    true variables with a specific tag, and returns a set of their base
    variable names.
    """
    tagged_identifiers = decode_model(model, predicate=tagger.is_tagged)
    return {tagger.strip_tag(identifier) for identifier in tagged_identifiers}


def group_constraints(
        tagger: Tagger,
        fbas: FBASGraph,
        group_by: str) -> list[Formula]:
    """
    Returns constraints that express that a group is tagged iff any of its
    members is tagged, and if a group is tagged, all its members are tagged.
    This makes all members of a group a single tagged unit.
    Also returns the set of groups.
    """
    constraints: list[Formula] = []
    groups = fbas.groups_dict(group_by)
    for group_name, members in groups.items():
        if members:
            # If group is tagged, all members are tagged
            constraints.append(
                Implies(tagger.atom(group_name),
                        And(*[tagger.atom(v) for v in members])))
            # If any member is tagged, group is tagged
            constraints.append(
                Implies(Or(*[tagger.atom(v) for v in members]),
                        tagger.atom(group_name)))
    return constraints, set(groups.keys())


def solve_constraints(constraints: list[Formula]) -> "slv.SatResult":
    clauses = to_cnf(constraints)
    return slv.solve_sat(clauses)


def quorum_constraints(fbas: FBASGraph,
                       make_atom: Callable[[str], Atom]) -> list[Formula]:
    """Returns constraints expressing that the set of true atoms is a quorum"""
    constraints: list[Formula] = []
    for v in fbas.vertices():
        if fbas.threshold(v) > 0:
            vs = [make_atom(n) for n in fbas.graph.successors(v)]
            constraints.append(
                Implies(
                    make_atom(v),
                    Card(
                        fbas.threshold(v),
                        *vs)))
        if fbas.graph.out_degree(v) == 0 or fbas.threshold(v) == 0:
            # TODO seems like this needs to be configurable; maybe a global
            # parameter to indicate how cautious we want to be.  Or maybe warn
            # after analysis when a result depends on this behavior
            continue
    # also require that the quorum contain at least one validator for which we
    # have a qset:
    constraints += [Or(*[make_atom(v)
                         for v in fbas.validators
                         if fbas.graph.out_degree(v) > 0])]
    return constraints


def contains_quorum(s: set[str], fbas: FBASGraph) -> bool:
    """
    Check if s contains a quorum.
    """
    assert s <= fbas.validators
    constraints: list[Formula] = []
    # the quorum constraints are satisfied:
    constraints += quorum_constraints(fbas, Atom)
    # no validators outside s are in the quorum:
    constraints += [And(*[Not(Atom(v)) for v in fbas.validators - s])]

    sat_res = solve_constraints(constraints)
    if sat_res.sat:
        q = decode_model(
            sat_res.model,
            predicate=lambda ident: ident in fbas.validators)
        logging.info("Quorum %s is inside %s", q, s)
    else:
        logging.info("No quorum found in %s", s)
    return sat_res.sat


def find_disjoint_quorums(
        fbas: FBASGraph) -> Optional[Tuple[Collection, Collection]]:
    """
    Find two disjoint quorums in the FBAS graph, or prove there are none.  To do
    this, we build a propositional formula that is satsifiable if and only if
    there are two disjoint quorums.  Then we call a SAT solver to check for
    satisfiability.

    The idea is to consider two quorums A and B and create two propositional
    variables vA and vB for each validator v, where vA is true when v is in
    quorum A, and vB is true when v is in quorum B.  Then we create constraints
    asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for
    satisfiability.  If the constraints are satisfiable, then we have two
    disjoint quorums and the truth assignment gives us the quorums.  Otherwise,
    we know that no two disjoint quorums exist.
    """
    taggers = {'A': Tagger("quorum_A"), 'B': Tagger("quorum_B")}

    def in_quorum(q, v) -> Atom:
        return taggers[q].atom(v)

    constraints: list[Formula] = []
    with timed("Disjoint-quorum constraint building"):
        for q in ['A', 'B']:  # our two quorums
            constraints += quorum_constraints(fbas, partial(in_quorum, q))
        # no validator can be in both quorums:
        for v in fbas.validators:
            constraints += [Or(Not(in_quorum('A', v)),
                               Not(in_quorum('B', v)))]

    with timed("CNF conversion"):
        clauses = to_cnf(constraints)

    sat_res = solve_constraints(constraints)
    res = sat_res.sat

    if config.output:
        if config.output:
            with open(config.output, 'w', encoding='utf-8') as f:
                cnf = CNF(from_clauses=clauses)
                dimacs = cnf.to_dimacs()
                comment = "c " + \
                    ("SATISFIABLE" if res else "UNSATISFIABLE") + "\n"
                f.write(comment)
                f.writelines(dimacs)

    if res:
        model = sat_res.model
        assert model  # error if []
        q1 = [
            v for v in extract_true_tagged_variables(
                model,
                taggers['A']) if v in fbas.validators]
        q2 = [
            v for v in extract_true_tagged_variables(
                model,
                taggers['B']) if v in fbas.validators]
        logging.info("Disjoint quorums found")
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1, over_approximate=True)
        assert fbas.is_quorum(q2, over_approximate=True)
        assert not set(q1) & set(q2)
        return (q1, q2)
    return None


def find_minimal_splitting_set(
        fbas: FBASGraph) -> Optional[Tuple[Collection, Collection, Collection]]:
    """
    Find a minimal-cardinality splitting set in the FBAS graph, or prove there
    is none.  Uses one of pysat's MaxSAT procedures (LSU or RC2).  If found,
    returns the splitting set and the two quorums that it splits.
    """

    logging.info(
        "Finding minimal-cardinality splitting set using MaxSAT algorithm %s with %s cardinality encoding",
        config.max_sat_algo,
        config.card_encoding)

    faulty_tagger = Tagger("faulty")
    quorum_a_tagger = Tagger("quorum_A")
    quorum_b_tagger = Tagger("quorum_B")
    quorum_taggers = {'A': quorum_a_tagger, 'B': quorum_b_tagger}

    constraints: list[Formula] = []
    with timed("Splitting-set constraint building"):
        # now we create the constraints:
        for q in ['A', 'B']:  # for each of our two quorums
            tagger = quorum_taggers[q]
            # the quorum contains at least one non-faulty validator for which we
            # have a qset:
            constraints += [Or(*[And(tagger.atom(n),
                                     Not(faulty_tagger.atom(n)))
                               for n in fbas.validators if fbas.threshold(n) >= 0])]
            # then, we add the threshold constraints:
            for v in fbas.vertices():
                if fbas.threshold(v) > 0:
                    vs = [tagger.atom(n) for n in fbas.graph.successors(v)]
                    if v in fbas.validators:
                        # the threshold must be met only if the validator is not
                        # faulty:
                        constraints.append(
                            Implies(And(tagger.atom(v),
                                        Not(faulty_tagger.atom(v))),
                                    Card(fbas.threshold(v),
                                         *vs)))
                    else:
                        # the threshold must be met:
                        constraints.append(
                            Implies(
                                tagger.atom(v),
                                Card(
                                    fbas.threshold(v),
                                    *vs)))
                if fbas.threshold(v) == 0:
                    continue  # no constraints for this vertex
        # add the constraint that no non-faulty validator can be in both
        # quorums:
        for v in fbas.validators:
            constraints += [Or(faulty_tagger.atom(v),
                               Not(quorum_a_tagger.atom(v)),
                               Not(quorum_b_tagger.atom(v)))]

        if config.group_by:
            constraints += group_constraints(
                faulty_tagger, fbas, config.group_by)

        # finally, convert to weighted CNF and add soft constraints that
        # minimize the number of faulty validators (or groups):
        wcnf = WCNF()
        wcnf.extend(to_cnf(constraints))
        if not config.group_by:
            for v in fbas.validators:
                wcnf.append(to_cnf(Not(faulty_tagger.atom(v)))[0], weight=1)
        else:
            for g in fbas.groups_dict(config.group_by).keys():  # type: ignore
                wcnf.append(to_cnf(Not(faulty_tagger.atom(g)))[0], weight=1)

    result = slv.solve_maxsat(wcnf)

    if not result.sat:
        logging.info("No splitting set found!")
        return None
    else:
        cost = result.optimum
        model = result.model
        logging.info(
            "Found minimal-cardinality splitting set of size is %s:",
            cost)
        model = list(model)
        ss = list(extract_true_tagged_variables(model, faulty_tagger))
        if not config.group_by:
            logging.info("Minimal-cardinality splitting set: %s",
                         [fbas.with_name(s) for s in ss])
        else:
            logging.info("Minimal-cardinality splitting set (groups): %s",
                         [s for s in ss if s in groups])  # type: ignore
            logging.info(
                "Minimal-cardinality splitting set (corresponding validators): %s",
                [fbas.with_name(s) for s in ss if s not in groups])  # type: ignore
        q1 = [
            v for v in extract_true_tagged_variables(
                model,
                quorum_a_tagger) if v in fbas.validators]
        q2 = [
            v for v in extract_true_tagged_variables(
                model,
                quorum_b_tagger) if v in fbas.validators]
        assert fbas.is_quorum(
            q1,
            over_approximate=True,
            no_requirements=set(ss))
        assert fbas.is_quorum(
            q2,
            over_approximate=True,
            no_requirements=set(ss))
        logging.info("Quorum A: %s", [fbas.with_name(v) for v in q1])
        logging.info("Quorum B: %s", [fbas.with_name(v) for v in q2])
        if not config.group_by:
            return (ss, q1, q2)
        else:
            return ([s for s in ss if s in groups], q1, q2)  # type: ignore


def find_minimal_blocking_set(fbas: FBASGraph) -> Optional[Collection[str]]:
    """
    Find a minimal-cardinality blocking set in the FBAS graph, or prove there is
    none.

    This is a bit more tricky than for splitting sets because we need to ensure
    that the "blocked-by" relation is well-founded (i.e. not circular). We
    achieve this by introducing a partial order on vertices and asserting that a
    vertex can only be blocked by vertices that are strictly lower in the order.
    """

    logging.info(
        "Finding minimal-cardinality blocking set using MaxSAT algorithm %s with %s cardinality encoding",
        config.max_sat_algo,
        config.card_encoding)

    if not fbas.validators:
        logging.info("No validators in the FBAS graph!")
        return None

    constraints: list[Formula] = []
    with timed("Blocking-set constraint building"):
        faulty_tagger = Tagger("faulty")
        blocked_tagger = Tagger("blocked")

        def lt(v1: str, v2: str) -> Formula:
            """
            v1 is strictly lower than v2
            """
            return Atom((v1, v2))

        def blocking_threshold(v) -> int:
            return len(list(fbas.graph.successors(v))) - fbas.threshold(v) + 1

        # first, the threshold constraints:
        for v in fbas.vertices():
            if v in fbas.validators:
                constraints.append(Or(faulty_tagger.atom(v),
                                      blocked_tagger.atom(v)))
            if v not in fbas.validators:
                constraints.append(Not(faulty_tagger.atom(v)))
            if fbas.threshold(v) > 0:
                may_block = [And(Or(blocked_tagger.atom(n),
                                    faulty_tagger.atom(n)),
                                 lt(n,
                                    v)) for n in fbas.graph.successors(v)]
                constraints.append(
                    Implies(
                        Card(
                            blocking_threshold(v),
                            *may_block),
                        blocked_tagger.atom(v)))
                constraints.append(
                    Implies(And(blocked_tagger.atom(v),
                                Not(faulty_tagger.atom(v))),
                            Card(blocking_threshold(v),
                                 *may_block)))
            if fbas.threshold(v) == 0:
                constraints.append(Not(blocked_tagger.atom(v)))

        # The lt relation must be a partial order (anti-symmetric and
        # transitive).  For performance, lt only relates vertices that are in
        # the same strongly connected components (as otherwise there is no
        # possible cycle anyway in the blocking relation).
        sccs = [scc for scc in nx.strongly_connected_components(fbas.graph)
                if any(fbas.threshold(v) >= 0 for v in set(scc))]
        assert sccs
        for scc in sccs:
            for v1 in scc:
                constraints.append(Not(lt(v1, v1)))
                for v2 in scc:
                    for v3 in scc:
                        constraints.append(
                            Implies(And(lt(v1, v2), lt(v2, v3)), lt(v1, v3)))

        groups = set()
        if config.group_by:
            constraints += group_constraints(
                faulty_tagger, fbas, config.group_by)

        # convert to weighted CNF and add soft constraints that minimize the
        # number of faulty validators:
        wcnf = WCNF()
        wcnf.extend(to_cnf(constraints))
        if not config.group_by:
            for v in fbas.validators:
                wcnf.append(to_cnf(Not(faulty_tagger.atom(v)))[0], weight=1)
        else:
            for g in fbas.groups_dict(config.group_by).keys():
                wcnf.append(to_cnf(Not(faulty_tagger.atom(g)))[0], weight=1)

    result = slv.solve_maxsat(wcnf)

    if not result.sat:
        logging.info("No blocking set found!")
        return None
    else:
        cost = result.optimum
        model = result.model
        model = list(model)
        logging.info(
            "Found minimal-cardinality blocking set, size is %s",
            cost)
        s = list(extract_true_tagged_variables(model, faulty_tagger))
        if not config.group_by:
            logging.info("Minimal-cardinality blocking set: %s",
                         [fbas.with_name(v) for v in s])
        else:
            logging.info("Minimal-cardinality blocking set: %s",
                         [g for g in s if g in groups])
        vs = set(s) - groups
        assert fbas.closure(vs) == fbas.validators
        for vs2 in combinations(vs, cost - 1):
            # TODO isn't this going to fail with groups?
            assert fbas.closure(vs2) != fbas.validators
        if not config.group_by:
            return s
        else:
            return [g for g in s if g in groups]


def min_history_loss_critical_set(
        fbas: FBASGraph) -> Tuple[Collection[str], Collection[str]]:
    """
    Return a set of minimal cardinality such that, should the validators in the
    set stop publishing valid history, the history may be lost.
    """

    logging.info(
        "Finding minimal-cardinality history-loss critical set using MaxSAT algorithm %s with %s cardinality encoding",
        config.max_sat_algo,
        config.card_encoding)

    constraints: list[Formula] = []

    in_critical_quorum_tagger = Tagger("in_critical_quorum")
    hist_error_tagger = Tagger("hist_error")
    in_crit_no_error_tagger = Tagger("in_crit_no_error")

    for v in fbas.validators:
        if fbas.vertice_attrs(v).get('historyArchiveHasError', True):
            constraints.append(hist_error_tagger.atom(v))
        else:
            constraints.append(Not(hist_error_tagger.atom(v)))

    for v in fbas.validators:
        constraints.append(
            Implies(
                in_critical_quorum_tagger.atom(v),
                Not(hist_error_tagger.atom(v)),
                in_crit_no_error_tagger.atom(v)))
        constraints.append(
            Implies(
                in_crit_no_error_tagger.atom(v),
                And(in_critical_quorum_tagger.atom(v),
                    Not(hist_error_tagger.atom(v)))))

    # the critical contains at least one validator for which we have a qset:
    constraints += [Or(*[in_critical_quorum_tagger.atom(v)
                       for v in fbas.validators if fbas.threshold(v) >= 0])]
    constraints += quorum_constraints(fbas, in_critical_quorum_tagger.atom)

    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # minimize the number of validators that are in the critical quorum but do
    # not have history errors:
    for v in fbas.validators:
        wcnf.append(to_cnf(Not(in_crit_no_error_tagger.atom(v)))[0], weight=1)

    result = slv.solve_maxsat(wcnf)

    if not result.sat:
        raise ValueError("No critical set found! This should not happen.")
    else:
        cost = result.optimum
        model = result.model
        logging.info(
            "Found minimal-cardinality history-critical set, size is %s",
            cost)
        model = list(model)
        min_critical = list(
            extract_true_tagged_variables(
                model, in_crit_no_error_tagger))
        quorum = [v for v in extract_true_tagged_variables(
            model, in_critical_quorum_tagger) if v in fbas.validators]
        logging.info("Minimal-cardinality history-critical set: %s",
                     [fbas.with_name(v) for v in min_critical])
        logging.info("Quorum: %s", [fbas.with_name(v) for v in quorum])
        return (min_critical, quorum)


def find_min_quorum(
        fbas: FBASGraph,
        not_subset_of=None,
        restrict_to_scc=True) -> Collection[str]:
    """
    Find a minimal quorum in the FBAS graph using pyqbf.  If not_subset_of is a
    set of validators, then the quorum should contain at least one validator
    outside this set.
    """

    if not HAS_QBF:
        raise ImportError(
            "QBF support not available. Install with: pip install python-fbas[qbf]")

    if restrict_to_scc:
        # First, find all sccs that contain at least one quorum:
        sccs = [scc for scc in nx.strongly_connected_components(fbas.graph)
                if any(fbas.threshold(v) > 0 for v in set(scc))]
        if not sccs:
            logging.info(
                "Found strongly connected components in the FBAS graph.")
            return []
        # Keep only the sccs that contain at least one quorum:
        sccs = [
            scc for scc in sccs if contains_quorum(
                set(scc) & fbas.validators, fbas)]
        if len(sccs) > 1:
            logging.warning("There are disjoint quorums")
        if len(sccs) == 0:
            logging.warning(
                "Found no SCC that contains a quorum. This should not happen!")
            return []

        # Now we have an scc that contains a quorum. Find a minimal quorum in
        # it.
        scc = sccs[0]
        fbas = fbas.project(scc & fbas.validators)

    if not fbas.validators:
        logging.info("The projected FBAS is empty!")
        return []

    quorum_a_tagger = Tagger("quorum_A")
    quorum_b_tagger = Tagger("quorum_B")

    def quorum_constraints_(q: str) -> list[Formula]:
        tagger = quorum_a_tagger if q == 'A' else quorum_b_tagger
        constraints: list[Formula] = []
        constraints += [Or(*[tagger.atom(n)
                           for n in fbas.validators if fbas.threshold(n) > 0])]
        constraints += quorum_constraints(fbas, tagger.atom)
        return constraints

    # The set 'A' is a quorum in the scc:
    qa_constraints: list[Formula] = quorum_constraints_('A')

    # it contains at least one validator outside not_subset_of:
    if not_subset_of:
        qa_constraints += [Or(*[quorum_a_tagger.atom(n)
                              for n in fbas.validators if n not in not_subset_of])]

    # If 'B' is a subset of 'A', then 'B' is not a quorum:
    qb_quorum = And(*quorum_constraints_('B'))
    qb_subset_qa = And(
        *[Implies(quorum_b_tagger.atom(n),
                  quorum_a_tagger.atom(n)) for n in fbas.validators])
    qb_constraints = Implies(qb_subset_qa, Not(qb_quorum))

    qa_clauses = to_cnf(qa_constraints)
    qb_clauses = to_cnf(qb_constraints)
    pcnf = PCNF(from_clauses=qa_clauses + qb_clauses)  # type: ignore

    qa_atoms: set[int] = atoms_of_clauses(qa_clauses)
    qb_vertex_atoms: set[int] = set(
        abs(variables[quorum_b_tagger.atom(n).identifier]) for n in fbas.vertices())
    qb_tseitin_atoms: set[int] = \
        atoms_of_clauses(qb_clauses) - (qb_vertex_atoms | qa_atoms)

    pcnf.exists(
        *list(qa_atoms)).forall(
            *list(qb_vertex_atoms)).exists(*list(qb_tseitin_atoms))

    qbf_res = slv.solve_qbf(pcnf)  # type: ignore
    res = qbf_res.sat
    if res:
        model = qbf_res.model
        qa = [
            v for v in extract_true_tagged_variables(
                model,
                quorum_a_tagger) if v in fbas.validators]
        assert fbas.is_quorum(qa, over_approximate=True)
        if not_subset_of:
            assert not set(qa) <= set(not_subset_of)
            if fbas.is_quorum(not_subset_of, over_approximate=False):
                assert not set(not_subset_of) <= set(qa)
        logging.info("Minimal quorum found: %s", qa)
        return qa
    else:
        logging.info("No minimal quorum found!")
        return []


def max_scc(fbas: FBASGraph) -> Collection[str]:
    """
    Compute a maximal strongly-connected component that contains a quorum
    """

    # First, find all sccs that contain at least one quorum:
    sccs = [scc for scc in nx.strongly_connected_components(fbas.graph)
            if any(fbas.threshold(v) > 0 for v in set(scc))]
    if not sccs:
        logging.info("Found strongly connected components in the FBAS graph.")
        return []
    # Keep only the sccs that contain at least one quorum:
    sccs = [
        scc for scc in sccs if contains_quorum(
            set(scc) & fbas.validators,
            fbas)]
    if len(sccs) > 1:
        logging.warning("There are disjoint quorums")
    if len(sccs) == 0:
        logging.warning(
            "Found no SCC that contains a quorum. This should not happen!")
        return []
    return {v for v in sccs[0] if v in fbas.validators}


def top_tier(fbas: FBASGraph) -> Collection[str]:
    """
    Compute the top tier of the FBAS graph, i.e. the union of all minimal quorums.
    """

    # First, find all sccs that contain at least one quorum:
    sccs = [scc for scc in nx.strongly_connected_components(fbas.graph)
            if any(fbas.threshold(v) > 0 for v in set(scc))]
    if not sccs:
        logging.info("Found strongly connected components in the FBAS graph.")
        return []
    # Keep only the sccs that contain at least one quorum:
    sccs = [scc for scc in sccs
            if contains_quorum(set(scc) & fbas.validators, fbas)]
    if len(sccs) > 1:
        logging.warning("There are disjoint quorums")
    if len(sccs) == 0:
        logging.warning(
            "Found no SCC that contains a quorum. This should not happen!")
        return []
    # Project onto one of the sccs:
    scc = sccs[0]
    fbas = fbas.project(scc & fbas.validators)

    top_tier_set: set[str] = set()
    while True:
        q = find_min_quorum(
            fbas,
            not_subset_of=top_tier_set,
            restrict_to_scc=False)
        if not q:
            break
        top_tier_set |= set(q)
    return top_tier_set


def is_overlay_resilient(fbas: FBASGraph, overlay: nx.Graph) -> bool:
    """
    Check if the overlay is FBA-resilient. That is, for every quorum Q, removing
    the complement of Q should not disconnect the overlay graph.
    """
    quorum_tagger = Tagger("in_quorum")
    constraints: list[Formula] = []
    # the quorum is non-empty (contains a validator with a valid qset):
    constraints += [Or(*[quorum_tagger.atom(v)
                       for v in fbas.validators if fbas.threshold(v) >= 0])]
    # the quorum constraints are satisfied:
    constraints += quorum_constraints(fbas, quorum_tagger.atom)

    # now we assert the graph is disconnected
    reachable_tagger = Tagger("reachable")
    # some node is reachable:
    constraints.append(Or(*[And(reachable_tagger.atom(v),
                                quorum_tagger.atom(v))
                       for v in fbas.validators]))
    # nodes outside the quorum are not reachable:
    constraints += [Implies(reachable_tagger.atom(v),
                            quorum_tagger.atom(v)) for v in fbas.validators]
    # for every two nodes in the quorum and with an edge between each other,
    # one is reachable iff the other is:
    constraints += [
        Implies(
            And(quorum_tagger.atom(v1),
                quorum_tagger.atom(v2)),
            equiv(reachable_tagger.atom(v1),
                  reachable_tagger.atom(v2)))
        for v1,
        v2 in overlay.edges() if v1 != v2]
    # some node in the quorum is unreachable:
    constraints.append(
        Or(*[And(Not(reachable_tagger.atom(v)),
                 quorum_tagger.atom(v)) for v in fbas.validators]))

    sat_res = solve_constraints(constraints)
    res = sat_res.sat
    if res:
        model = sat_res.model
        assert model  # error if []
        q = [
            v for v in extract_true_tagged_variables(
                model,
                quorum_tagger) if v in fbas.validators]
        logging.info("Quorum %s is disconnected", q)
    else:
        logging.info("The overlay is FBA-resilient!")
    return not res


def num_not_blocked(fbas: FBASGraph, overlay: nx.Graph) -> int:
    """
    Returns the number of validators v such that v is not blocked by its set of
    neighbors in the overlay.

    If this returns 0 then we know that, for every quorum Q, if we remove Q from
    the overlay graph, then each remaining validator still has at least one
    neighbor. Note that this does not imply that the graph remains connected.
    """
    n = 0
    for v in fbas.validators:
        peers = list(overlay.neighbors(v)) if v in overlay else []
        if v not in fbas.closure(peers):
            n += 1
    return n


def is_fba_resilient_approx(fbas: FBASGraph, overlay: nx.Graph) -> bool:
    """
    Check if every node is blocked by its set of neighbors in the overlay. Note
    this does not guarantee connectivity under maximal failures (i.e. removing
    the complement of a quorum).
    """
    for v in fbas.validators:
        peers = list(overlay.neighbors(v)) if v in overlay else []
        if v not in fbas.closure(peers):
            return False
    return True
