"""
SAT-based analysis of FBAS graphs
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Collection
from itertools import combinations
from functools import partial
import networkx as nx
from pysat.formula import CNF, WCNF
from python_fbas import solver as slv
from python_fbas.fbas_graph import FBASGraph
from python_fbas.propositional_logic import (
    And, Or, Implies, Atom, Formula, AtLeast, Not, equiv, variables, to_cnf,
    atoms_of_clauses, decode_model
)
import python_fbas.config as config
from python_fbas.utils import timed
try:
    from pyqbf.formula import PCNF
    HAS_QBF = True
except ImportError:
    HAS_QBF = False
try:
    from pysat.allies.unigen import Sampler as UniGenSampler
except ImportError:
    UniGenSampler = None

@dataclass
class DisjointQuorumsResult:
    """Represents the result of finding disjoint quorums."""
    quorum_a: Collection[str]
    quorum_b: Collection[str]


@dataclass
class SplittingSetResult:
    """Represents the result of finding a minimal splitting set."""
    splitting_set: Collection[str]
    quorum_a: Collection[str]
    quorum_b: Collection[str]


@dataclass
class HistoryLossResult:
    """Represents the result of finding a minimal history loss critical set."""
    min_critical_set: Collection[str]
    quorum: Collection[str]


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
                                  tagger: Tagger) -> set[Any]:
    """
    Given a pysat model (a list of ints), it decodes the model, filters for
    true variables with a specific tag, and returns a set of their base
    variable names.
    """
    tagged_identifiers = decode_model(model, predicate=tagger.is_tagged)
    return {tagger.strip_tag(identifier) for identifier in tagged_identifiers}


def solve_constraints(constraints: list[Formula]) -> "slv.SatResult":
    clauses = to_cnf(constraints)
    return slv.solve_sat(clauses)


def quorum_constraints(fbas: FBASGraph,
                       make_atom: Callable[[str], Atom],
                       *,
                       over_approximate: bool = True) -> list[Formula]:
    """Returns constraints expressing that the set of true atoms is a quorum"""
    constraints: list[Formula] = []
    for q in fbas.get_qset_vertices():
        assert fbas.threshold(q) is not None
        if fbas.threshold(q) > 0:
            vs = [make_atom(n) for n in fbas.graph_view().successors(q)]
            constraints.append(
                Implies(
                    make_atom(q),
                    AtLeast(fbas.threshold(q), *vs)))
    for v in fbas.get_validators():
        qset = fbas.qset_vertex_of(v)
        if qset:
            constraints.append(Implies(make_atom(v), make_atom(qset)))
        elif not over_approximate:
            constraints.append(Not(make_atom(v)))
    # also require that the quorum contain at least one validator for which we
    # have a qset:
    constraints += [Or(*[make_atom(v)
                         for v in fbas.get_validators()
                         if fbas.qset_vertex_of(v)])]
    return constraints

def not_quorum_constraint(fbas: FBASGraph,
                           make_atom: Callable[[str], Atom],
                           ) -> Formula:
    """
    Returns constraints expressing that the set of true atoms is _not_ a quorum.

    We need this because we cannot negate totalizer-encoded cardinality constraints.
    """
    qset_not_sat: list[Formula] = []
    for q in fbas.get_qset_vertices():
        assert fbas.threshold(q) is not None
        if fbas.threshold(q) > 0:
            vs = [Not(make_atom(n)) for n in fbas.graph_view().successors(q)]
            t = fbas.graph_view().out_degree(q) - fbas.threshold(q) + 1
            qset_not_sat.append(And(make_atom(q), AtLeast(t, *vs)))
    validator_not_sat: list[Formula] = []
    for v in fbas.get_validators():
        qset = fbas.qset_vertex_of(v)
        if qset:
            validator_not_sat.append(And(make_atom(v), Not(make_atom(qset))))
    is_empty = [And(*[Not(make_atom(v)) for v in fbas.get_validators() if fbas.qset_vertex_of(v)])]
    disj = qset_not_sat + validator_not_sat + is_empty
    return Or(*disj)

def group_constraints(
        tagger: Tagger,
        fbas: FBASGraph,
        group_by: str) -> list[Formula]:
    """
    Returns constraints that express that a tagged group atom is true iff any
    tagged validator atom of its members is true.
    """
    constraints: list[Formula] = []
    groups = fbas.groups_dict(group_by)
    for group_name, members in groups.items():
        if members:
            group_id = group_var(group_name)
            member_atoms = [tagger.atom(v) for v in members]
            # Group is tagged iff any member is tagged.
            constraints.append(
                Implies(tagger.atom(group_id), Or(*member_atoms)))
            constraints.append(
                Implies(Or(*member_atoms), tagger.atom(group_id)))
    return constraints


def group_var(name: str) -> tuple[str, str]:
    return ("group", name)


def contains_quorum(s: set[str], fbas: FBASGraph) -> bool:
    """
    Check if s contains a quorum.
    """
    assert s <= fbas.get_validators()
    constraints: list[Formula] = []
    # the quorum constraints are satisfied:
    constraints += quorum_constraints(fbas, Atom)
    # no validators outside s are in the quorum:
    constraints += [And(*[Not(Atom(v)) for v in fbas.get_validators() - s])]

    sat_res = solve_constraints(constraints)
    if sat_res.sat:
        q = decode_model(
            sat_res.model,
            predicate=lambda ident: fbas.is_validator(ident))
        logging.info("Quorum\n%s\n is inside\n %s", q, s)
        missing_qset = [v for v in q if fbas.graph_view().out_degree(v) == 0]
        if missing_qset:
            logging.warning(f"validators {missing_qset} do not have a qset")
        assert fbas.is_quorum(q)
    else:
        logging.info("No quorum found in %s", s)
    return sat_res.sat


def random_quorum(
        fbas: FBASGraph,
        *,
        seed: int | None = None,
        epsilon: float = 0.8,
        delta: float = 0.2,
        kappa: float = 0.638) -> Collection[str] | None:
    """
    Return a random quorum using UniGen's sampler, or None if no quorum exists.
    """
    if UniGenSampler is None:
        raise ImportError(
            "UniGen support not available. Install pysat with UniGen dependencies.")

    if not fbas.get_validators():
        logging.info("The FBAS has no validators!")
        return None

    constraints = quorum_constraints(fbas, Atom)
    clauses = to_cnf(constraints)
    sat_res = slv.solve_sat(clauses, label="Quorum SAT check")
    if not sat_res.sat:
        logging.info("No quorum found in the FBAS graph.")
        return None

    sample_over = [
        abs(variables[Atom(v).identifier])
        for v in fbas.get_validators()
        if Atom(v).identifier in variables
    ]
    if not sample_over:
        logging.info("No validators with qsets available for sampling.")
        return None

    if seed is None:
        seed = random.SystemRandom().randint(1, 2**31 - 1)

    cnf = CNF(from_clauses=clauses)
    try:
        with UniGenSampler(formula=cnf, seed=seed, epsilon=epsilon,
                           delta=delta, kappa=kappa) as sampler:
            samples = sampler.sample(nof_samples=1, sample_over=sample_over)
    except AssertionError as exc:
        raise ImportError(
            "UniGen support not available. Install pyunigen.") from exc

    if not samples:
        logging.info("UniGen returned no samples.")
        return None

    sample = samples[0]
    quorum = {v for v in decode_model(sample, predicate=fbas.is_validator)}
    return quorum


def find_disjoint_quorums(
        fbas: FBASGraph) -> DisjointQuorumsResult | None:
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

    def in_quorum(q: str, v: str) -> Atom:
        return taggers[q].atom(v)

    constraints: list[Formula] = []
    with timed("Disjoint-quorum constraint building"):
        for q in ['A', 'B']:  # our two quorums
            constraints += quorum_constraints(fbas, partial(in_quorum, q))
        # no validator can be in both quorums:
        for v in fbas.get_validators():
            constraints += [Or(Not(in_quorum('A', v)),
                               Not(in_quorum('B', v)))]

    with timed("CNF conversion"):
        clauses = to_cnf(constraints)

    sat_res = solve_constraints(constraints)
    res = sat_res.sat

    if config.get().output:
        output_file = config.get().output
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                cnf = CNF(from_clauses=clauses)
                dimacs = cnf.to_dimacs()
                comment = "c " + ("SATISFIABLE" if res else "UNSATISFIABLE") \
                    + "\n"
                f.write(comment)
                f.writelines(dimacs)

    if res:
        model = sat_res.model
        q1 = [
            v for v in extract_true_tagged_variables(
                model,
                taggers['A']) if fbas.is_validator(v)]
        q2 = [
            v for v in extract_true_tagged_variables(
                model,
                taggers['B']) if fbas.is_validator(v)]
        logging.info("Disjoint quorums found")
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1, over_approximate=True)
        assert fbas.is_quorum(q2, over_approximate=True)
        assert not set(q1) & set(q2)
        missing_qset = [v for v in set(q1) | set(q2) if fbas.graph_view().out_degree(v) == 0]
        if missing_qset:
            logging.warning(f"validators {missing_qset} do not have a qset")
        return DisjointQuorumsResult(quorum_a=q1, quorum_b=q2)
    return None


def find_minimal_splitting_set(
        fbas: FBASGraph) -> SplittingSetResult | None:
    """
    Find a minimal-cardinality splitting set in the FBAS graph, or prove there
    is none.  Uses one of pysat's MaxSAT procedures (LSU or RC2).  If found,
    returns the splitting set and the two quorums that it splits.
    """

    logging.info(
        "Finding minimal-cardinality splitting set using MaxSAT algorithm %s with %s cardinality encoding",
        config.get().max_sat_algo,
        config.get().card_encoding)

    faulty_tagger = Tagger("faulty")
    quorum_a_tagger = Tagger("quorum_A")
    quorum_b_tagger = Tagger("quorum_B")
    quorum_taggers = {'A': quorum_a_tagger, 'B': quorum_b_tagger}

    def is_faulty(v: Any) -> Atom:
        return faulty_tagger.atom(v)

    def in_quorum(q: str, v: Any) -> Atom:
        return quorum_taggers[q].atom(v)

    constraints: list[Formula] = []
    with timed("Splitting-set constraint building"):
        # now we create the constraints:
        for q in ['A', 'B']:  # for each of our two quorums
            # the quorum contains at least one non-faulty validator for which we
            # have a qset:
            constraints += \
                [Or(*[And(in_quorum(q, n), Not(is_faulty(n)))
                      for n in fbas.get_validators()
                      if fbas.has_qset(n)])]
            # then, we add the threshold constraints:
            for v in fbas.vertices():
                if fbas.is_validator(v) and fbas.has_qset(v):
                    # non-faulty validators must have their requirements met:
                    constraints.append(
                        Implies(*[in_quorum(q, v), Not(is_faulty(v)),
                                  in_quorum(q, fbas.qset_vertex_of(v))]))
                elif not fbas.is_validator(v) and fbas.threshold(v) > 0:
                    member_atoms = [in_quorum(q, n)
                                    for n in fbas.graph_view().successors(v)]
                    # the threshold must be met:
                    constraints.append(
                        Implies(
                            in_quorum(q, v),
                            AtLeast(fbas.threshold(v), *member_atoms)))
        # add the constraint that no non-faulty validator can be in both
        # quorums:
        for v in fbas.get_validators():
            constraints += [Or(is_faulty(v),
                               Not(in_quorum('A', v)),
                               Not(in_quorum('B', v)))]

        groups: set[tuple[str, str]] = set()
        group_lookup: dict[tuple[str, str], str] = {}

        group_by = config.get().group_by
        if group_by:
            constraints += group_constraints(faulty_tagger, fbas, group_by)
            group_names = fbas.groups_dict(group_by).keys()
            groups = {group_var(name) for name in group_names}
            group_lookup = {group_var(name): name for name in group_names}

        # finally, convert to weighted CNF and add soft constraints that
        # minimize the number of faulty validators (or groups):
        wcnf = WCNF()
        wcnf.extend(to_cnf(constraints))
        if not config.get().group_by:
            for v in fbas.get_validators():
                wcnf.append(to_cnf(Not(is_faulty(v)))[0], weight=1)
        else:
            for g in groups:  # type: ignore
                wcnf.append(to_cnf(Not(is_faulty(g)))[0], weight=1)

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
        ss = extract_true_tagged_variables(model, faulty_tagger)
        if not config.get().group_by:
            logging.info("Minimal-cardinality splitting set: %s",
                         [fbas.format_validator(s) for s in ss])
        else:
            logging.info("Minimal-cardinality splitting set (groups): %s",
                         [group_lookup[s] for s in ss if s in groups])
            logging.info(
                "Minimal-cardinality splitting set (corresponding validators): %s",
                [fbas.format_validator(s) for s in ss if s not in groups])

        # TODO: it would be nice to "minimize" q1 and q2
        q1 = {v for v in extract_true_tagged_variables(model, quorum_a_tagger)
            if fbas.is_validator(v)}
        q2 = {v for v in extract_true_tagged_variables(model, quorum_b_tagger)
              if fbas.is_validator(v)}
        assert all(fbas.is_sat(v, q1, over_approximate=True) for v in q1 - ss)
        assert all(fbas.is_sat(v, q2, over_approximate=True) for v in q2 - ss)
        logging.info("Quorum A: %s", [fbas.format_validator(v) for v in q1])
        logging.info("Quorum B: %s", [fbas.format_validator(v) for v in q2])
        missing_qset = [v for v in q1 | q2 if fbas.graph_view().out_degree(v) == 0]
        if missing_qset:
            logging.warning(f"validators {missing_qset} do not have a qset")
        if not config.get().group_by:
            return SplittingSetResult(splitting_set=ss, quorum_a=q1, quorum_b=q2)
        else:
            return SplittingSetResult(
                splitting_set=[group_lookup[s] for s in ss if s in groups],
                quorum_a=q1,
                quorum_b=q2)


def sccs_including_quorum(fbas: FBASGraph) -> list[Collection[str]]:
    # First, find all sccs that contain at least one quorum:
    sccs = [scc for scc in nx.strongly_connected_components(fbas.graph_view())
            if any(fbas.has_qset(v) for v in set(scc) & fbas.get_validators())]
    if not sccs:
        logging.info("Found no strongly connected components in the FBAS graph.")
        return []
    # Keep only the sccs that contain at least one quorum:
    sccs = [scc for scc in sccs
            if contains_quorum(set(scc) & fbas.get_validators(), fbas)]
    if len(sccs) > 1:
        logging.warning("There are disjoint quorums")
    if len(sccs) == 0:
        logging.warning("Found no SCC that contains a quorum. This is due to validators for which we do not have a qset.")
    return sccs


def max_scc(fbas: FBASGraph) -> Collection[str]:
    """
    Compute a maximal strongly-connected component that contains a quorum
    """
    sccs = sccs_including_quorum(fbas)
    if sccs:
        return {v for v in sccs[0] if fbas.is_validator(v)}
    else:
        return {}


def find_minimal_blocking_set(fbas: FBASGraph) -> Collection[str] | None:
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
        config.get().max_sat_algo,
        config.get().card_encoding)

    if not fbas.get_validators():
        logging.info("No validators in the FBAS graph!")
        return None

    constraints: list[Formula] = []
    with timed("Blocking-set constraint building"):
        faulty_tagger = Tagger("faulty")
        blocked_tagger = Tagger("blocked")
        lt_tagger = Tagger("lt")

        def is_faulty(v: Any) -> Atom:
            return faulty_tagger.atom(v)

        def is_blocked(v: Any) -> Atom:
            return blocked_tagger.atom(v)

        def lt(v1: str, v2: str) -> Formula:
            """
            v1 is strictly lower than v2 in the "blocked-by" order
            """
            return lt_tagger.atom((v1, v2))

        def blocking_threshold(v: str) -> int:
            assert not fbas.is_validator(v)
            return len(list(fbas.graph_view().successors(v))) - fbas.threshold(v) + 1

        # first, the threshold constraints:
        for v in fbas.get_validators():
            constraints.append(Or(is_faulty(v), is_blocked(v)))
            if fbas.graph_view().out_degree(v) == 0:
                constraints.append(
                    equiv(
                        Or(*[is_faulty(v2) for v2 in fbas.get_validators()]),
                        is_blocked(v)))  # be conservative
            else:
                assert fbas.graph_view().out_degree(v) == 1
                q = fbas.qset_vertex_of(v)
                constraints.append(
                    equiv(
                        And(is_blocked(q), lt(q, v)),
                        is_blocked(v)))

        for q in fbas.get_qset_vertices():
            constraints.append(Not(is_faulty(q)))
            if fbas.threshold(q) == 0:
                constraints.append(Not(is_blocked(q)))
            else:
                assert fbas.threshold(q) > 0
                vs = set(fbas.graph_view().successors(q)) & fbas.get_validators()
                qs = {q for q in fbas.graph_view().successors(q)
                      if q in fbas.get_qset_vertices()}
                may_block = [And(Or(is_blocked(n), is_faulty(n)),
                                 lt(n, q))
                             for n in vs]
                may_block += [And(is_blocked(n), lt(n, q)) for n in qs]
                constraints.append(
                    # one direction should be sufficient since we'll be minimizing
                    Implies(
                        is_blocked(q),
                        AtLeast(blocking_threshold(q), *may_block)))

        # The lt relation must be a partial order (anti-symmetric and
        # transitive).  For performance, lt only relates vertices that are in
        # the same strongly connected components (as otherwise there is no
        # possible cycle anyway in the blocking relation).
        sccs = nx.strongly_connected_components(fbas.graph_view())
        assert sccs
        for scc in sccs:
            for v1 in scc:
                constraints.append(Not(lt(v1, v1)))
                for v2 in scc - {v1}:
                    for v3 in scc - {v2}:
                        constraints.append(
                            Implies(*[lt(v1, v2), lt(v2, v3), lt(v1, v3)]))
        # for v1 in fbas.vertices():
        #     constraints.append(Not(lt(v1, v1)))
        #     for v2 in fbas.vertice():
        #         for v3 in fbas.vertices():
        #             constraints.append(
        #                 Implies(And(lt(v1, v2), lt(v2, v3)), lt(v1, v3)))

        groups: set[tuple[str, str]] = set()
        group_lookup: dict[tuple[str, str], str] = {}
        group_by = config.get().group_by
        if group_by:
            constraints += group_constraints(
                faulty_tagger, fbas, group_by)
            group_names = fbas.groups_dict(group_by).keys()
            groups = {group_var(name) for name in group_names}
            group_lookup = {group_var(name): name for name in group_names}

        # convert to weighted CNF and add soft constraints that minimize the
        # number of faulty validators:
        wcnf = WCNF()
        wcnf.extend(to_cnf(constraints))
        if not config.get().group_by:
            for v in fbas.get_validators():
                wcnf.append(to_cnf(Not(is_faulty(v)))[0], weight=1)
        else:
            for g in groups:
                wcnf.append(to_cnf(Not(is_faulty(g)))[0], weight=1)

    result = slv.solve_maxsat(wcnf)

    if not result.sat:
        logging.info("No blocking set found!")
        return None
    else:
        cost = result.optimum
        if cost is None:
            logging.error("No optimal cost found")
            assert False
        model = result.model
        model = list(model)
        logging.info(
            "Found minimal-cardinality blocking set, size is %s",
            cost)
        s = list(extract_true_tagged_variables(model, faulty_tagger))
        if not config.get().group_by:
            logging.info("Minimal-cardinality blocking set: %s",
                         [fbas.format_validator(v) for v in s])
        else:
            logging.info("Minimal-cardinality blocking set: %s",
                         [group_lookup[g] for g in s if g in groups])
        vs = set(s) - groups
        no_qset = {v for v in fbas.get_validators() if fbas.graph_view().out_degree(v) == 0}
        assert fbas.closure(vs | no_qset) == fbas.get_validators()
        if not fbas.closure(vs) | no_qset == fbas.get_validators():
            logging.warning(f"The validators {no_qset} have no known qset and this affects the blocking-set analysis results")
        # run some checks if it's cheap enough:
        if cost > 0 and len(list(vs)) < 4 and not config.get().group_by:
            for vs2 in combinations(vs, cost - 1):
                assert fbas.closure(vs2) != fbas.get_validators()
        if not config.get().group_by:
            return s
        else:
            return [group_lookup[g] for g in s if g in groups]


def min_history_loss_critical_set(
        fbas: FBASGraph) -> HistoryLossResult:
    """
    Return a set of minimal cardinality such that, should the validators in the
    set stop publishing valid history, the history may be lost.
    """

    logging.info(
        "Finding minimal-cardinality history-loss critical set using MaxSAT algorithm %s with %s cardinality encoding",
        config.get().max_sat_algo,
        config.get().card_encoding)

    constraints: list[Formula] = []

    in_critical_quorum_tagger = Tagger("in_critical_quorum")
    hist_error_tagger = Tagger("hist_error")
    in_crit_no_error_tagger = Tagger("in_crit_no_error")

    def in_critical_quorum(v: Any) -> Atom:
        return in_critical_quorum_tagger.atom(v)

    def has_hist_error(v: Any) -> Atom:
        return hist_error_tagger.atom(v)

    def in_crit_no_error(v: Any) -> Atom:
        return in_crit_no_error_tagger.atom(v)

    for v in fbas.get_validators():
        if fbas.vertice_attrs(v).get('historyArchiveHasError', True):
            constraints.append(has_hist_error(v))
        else:
            constraints.append(Not(has_hist_error(v)))

    for v in fbas.get_validators():
        constraints.append(
            Implies(
                in_critical_quorum(v),
                Not(has_hist_error(v)),
                in_crit_no_error(v)))
        constraints.append(
            Implies(
                in_crit_no_error(v),
                And(in_critical_quorum(v), Not(has_hist_error(v)))))

    # the critical contains at least one validator for which we have a qset:
    constraints += [Or(*[in_critical_quorum(v)
                       for v in fbas.get_validators() if fbas.has_qset(v)])]
    constraints += quorum_constraints(fbas, in_critical_quorum)

    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # minimize the number of validators that are in the critical quorum but do
    # not have history errors:
    for v in fbas.get_validators():
        wcnf.append(to_cnf(Not(in_crit_no_error(v)))[0], weight=1)

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
            model, in_critical_quorum_tagger) if fbas.is_validator(v)]
        logging.info("Minimal-cardinality history-critical set: %s",
                     [fbas.format_validator(v) for v in min_critical])
        logging.info("Quorum: %s", [fbas.format_validator(v) for v in quorum])
        return HistoryLossResult(min_critical_set=min_critical, quorum=quorum)


def find_min_quorum(
        fbas: FBASGraph,
        *,
        not_subset_of: Collection[str] | None = None,
        project_on_scc: bool = True,
        cardinality: int | None = None) -> Collection[str]:
    """
    Find a minimal quorum in the FBAS graph using pyqbf.  If not_subset_of is a
    set of validators, then the quorum should contain at least one validator
    outside this set. If cardinality is set, require exactly that many
    validators in the quorum.
    """
    if not HAS_QBF:
        raise ImportError(
            "QBF support not available. Install with: pip install python-fbas[qbf]")

    if project_on_scc:
        sccs = sccs_including_quorum(fbas)
        if not sccs:
            return []
        scc = set(sccs[0])
        fbas = fbas.project_on_reachable_from(scc & fbas.get_validators())

    if not fbas.get_validators():
        logging.info("The FBAS has no validators!")
        return []
    validators = list(fbas.get_validators())

    quorum_a_tagger = Tagger("quorum_A")
    quorum_b_tagger = Tagger("quorum_B")

    def in_quorum_a(v: Any) -> Atom:
        return quorum_a_tagger.atom(v)

    def in_quorum_b(v: Any) -> Atom:
        return quorum_b_tagger.atom(v)

    # The set 'A' is a quorum in the scc:
    qa_constraints: list[Formula] = quorum_constraints(fbas, in_quorum_a)
    if cardinality is not None:
        if cardinality < 1 or cardinality > len(validators):
            raise ValueError(
                f"cardinality {cardinality} is outside 1..{len(validators)}")
        qa_constraints.append(
            AtLeast(cardinality, *[in_quorum_a(v) for v in validators]))
        if cardinality < len(validators):
            qa_constraints.append(
                AtLeast(
                    len(validators) - cardinality,
                    *[Not(in_quorum_a(v)) for v in validators]))

    # it contains at least one validator outside not_subset_of:
    if not_subset_of:
        qa_constraints += [Or(*[in_quorum_a(n)
                              for n in fbas.get_validators() if n not in not_subset_of])]

    # If 'B' is a strict subset of 'A', then 'B' is not a quorum:
    qb_not_quorum = not_quorum_constraint(fbas, in_quorum_b)
    qb_subset_qa_constraints = \
        [Implies(in_quorum_b(n), in_quorum_a(n))
         for n in fbas.get_validators()]
    # strict inclusion:
    qb_subset_qa_constraints += \
        [Or(*[And(in_quorum_a(n), Not(in_quorum_b(n)))
              for n in fbas.get_validators()])]
    qb_constraints = Implies(*qb_subset_qa_constraints, qb_not_quorum)

    qa_clauses = to_cnf(qa_constraints)
    qb_clauses = to_cnf(qb_constraints)
    pcnf = PCNF(from_clauses=qa_clauses + qb_clauses)  # type: ignore

    qa_atoms: set[int] = atoms_of_clauses(qa_clauses)
    qa_vertex_atoms: set[int] = set(
        abs(variables[in_quorum_a(n).identifier]) for n in fbas.vertices())
    qb_atoms: set[int] = atoms_of_clauses(qb_clauses)
    qb_vertex_atoms: set[int] = set(
        abs(variables[in_quorum_b(n).identifier]) for n in fbas.vertices())
    tseitin_atoms: set[int] = \
        (qa_atoms | qb_atoms) - (qb_vertex_atoms | qa_vertex_atoms)

    pcnf.exists(
        *list(qa_vertex_atoms)).forall(
            *list(qb_vertex_atoms)).exists(*list(tseitin_atoms))

    # exit(-1)
    qbf_res = slv.solve_qbf(pcnf)  # type: ignore
    res = qbf_res.sat
    if res:
        model = qbf_res.model
        qa = [
            v for v in extract_true_tagged_variables(
                model,
                quorum_a_tagger) if fbas.is_validator(v)]
        assert fbas.is_quorum(qa, over_approximate=True)
        if not_subset_of:
            assert not set(qa) <= set(not_subset_of)
        logging.info("Minimal quorum found: %s",  [fbas.format_validator(v) for v in qa])
        return qa
    else:
        logging.info("No minimal quorum found!")
        return []


def find_min_cardinality_min_quorum(
        fbas: FBASGraph,
        *,
        not_subset_of: Collection[str] | None = None,
        project_on_scc: bool = True) -> Collection[str]:
    """
    Find a minimal-cardinality minimal quorum in the FBAS graph using pyqbf.
    """
    if not HAS_QBF:
        raise ImportError(
            "QBF support not available. Install with: pip install python-fbas[qbf]")

    if project_on_scc:
        sccs = sccs_including_quorum(fbas)
        if not sccs:
            return []
        scc = set(sccs[0])
        fbas = fbas.project_on_reachable_from(scc & fbas.get_validators())

    if not fbas.get_validators():
        logging.info("The FBAS has no validators!")
        return []

    max_cardinality = len(fbas.get_validators())
    for cardinality in range(1, max_cardinality + 1):
        quorum = find_min_quorum(
            fbas,
            not_subset_of=not_subset_of,
            project_on_scc=False,
            cardinality=cardinality)
        if quorum:
            return quorum
    return []


def top_tier(fbas: FBASGraph, *, from_validator: str | None = None) -> Collection[str]:
    """
    Compute the what we call the top tier of the FBAS graph, i.e. the union of
    all minimal quorums.
    """

    if from_validator:
        if not fbas.is_validator(from_validator):
            raise ValueError(f"{from_validator} is not a known validator")
        fbas = fbas.restrict_to_reachable(from_validator)

    if find_disjoint_quorums(fbas):
        logging.warning("The FBAS contains disjoint quorums")
        return []

    sccs = sccs_including_quorum(fbas)
    if not sccs:
        return []
    scc = set(sccs[0])
    fbas = fbas.project_on_reachable_from(scc & fbas.get_validators())

    top_tier_set: set[str] = set()
    while True:
        q = find_min_quorum(fbas, not_subset_of=top_tier_set, project_on_scc=False)
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
    reachable_tagger = Tagger("reachable")

    def in_quorum(v: Any) -> Atom:
        return quorum_tagger.atom(v)

    def is_reachable(v: Any) -> Atom:
        return reachable_tagger.atom(v)

    constraints: list[Formula] = []
    # the quorum constraints are satisfied:
    constraints += quorum_constraints(fbas, in_quorum)

    # now we assert the graph is disconnected
    # some node is reachable:
    constraints.append(Or(*[And(is_reachable(v),
                                in_quorum(v))
                       for v in fbas.get_validators()]))
    # nodes outside the quorum are not reachable:
    constraints += [Implies(is_reachable(v), in_quorum(v))
                    for v in fbas.get_validators()]
    # for every two nodes in the quorum and with an edge between each other,
    # one is reachable iff the other is:
    constraints += [
        Implies(*[in_quorum(v1), in_quorum(v2),
                  equiv(is_reachable(v1), is_reachable(v2))])
        for v1, v2 in overlay.edges() if v1 != v2]
    # some node in the quorum is unreachable:
    constraints.append(
        Or(*[And(Not(is_reachable(v)),
                 in_quorum(v)) for v in fbas.get_validators()]))

    sat_res = solve_constraints(constraints)
    res = sat_res.sat
    if res:
        model = sat_res.model
        assert model  # error if []
        q = [
            v for v in extract_true_tagged_variables(
                model,
                quorum_tagger) if fbas.is_validator(v)]
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
    for v in fbas.get_validators():
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
    for v in fbas.get_validators():
        peers = list(overlay.neighbors(v)) if v in overlay else []
        if v not in fbas.closure(peers):
            return False
    return True
