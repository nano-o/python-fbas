"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Any, Optional, Tuple, Collection, Callable, Union
from itertools import combinations
import networkx as nx
from pysat.solvers import Solver, SolverNames
from pysat.formula import CNF, WCNF
from pysat.examples.lsu import LSU  # MaxSAT algorithm
from pysat.examples.rc2 import RC2  # MaxSAT algorithm
from python_fbas.fbas_graph import FBASGraph
from python_fbas.propositional_logic \
    import And, Or, Implies, Atom, Formula, Card, Not, equiv, variables, variables_inv, to_cnf, atoms_of_clauses
import python_fbas.config as config
try:
    from pyqbf.formula import PCNF
    from pyqbf.solvers import Solver as QSolver
    HAS_QBF = True
except ImportError:
    HAS_QBF = False

solvers: list[str] = [list(SolverNames.__dict__[s])[::-1][0]
                      for s in SolverNames.__dict__ if not s.startswith('__')]


def quorum_constraints(fbas: FBASGraph,
                       make_atom: Callable[[str],
                                           Atom]) -> list[Formula]:
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
        if fbas.threshold(v) == 0:
            continue # no constraints for this vertex
    return constraints


def solve_constraints(constraints: list[Formula]) -> Tuple[bool, Solver]:
    clauses = to_cnf(constraints)
    solver = Solver(bootstrap_with=clauses, name=config.sat_solver)
    start_time = time.time()
    res: bool = bool(solver.solve())
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    return res, solver


def contains_quorum(s: set[str], fbas: FBASGraph) -> bool:
    """
    Check if s contains a quorum.
    """
    assert s <= fbas.validators
    constraints: list[Formula] = []
    # the quorum must contain at least one validator from s (and for which we
    # have a qset):
    constraints += [Or(*[Atom(v) for v in s if fbas.threshold(v) >= 0])]
    # no validators outside s are in the quorum:
    constraints += [And(*[Not(Atom(v))
                        for v in fbas.validators if v not in s])]
    # and the quorum constraints are satisfied:
    constraints += quorum_constraints(fbas, Atom)

    res, solver = solve_constraints(constraints)
    if res:
        model: list[int] = solver.get_model() or []
        assert model  # error if []
        q = [variables_inv[v]
             for v in set(model) & set(variables_inv.keys())
             if variables_inv[v] in fbas.validators]
        logging.info("Quorum %s is inside %s", q, s)
    else:
        logging.info("No quorum found in %s", s)
    return res


def find_disjoint_quorums(
        fbas: FBASGraph) -> Optional[Tuple[Collection, Collection]]:
    """
    Find two disjoint quorums in the FBAS graph, or prove there are none.
    To do this, we build a propositional formula that is satsifiable if and only if there are two disjoint quorums.
    Then we call a SAT solver to check for satisfiability.

    The idea is to consider two quorums A and B and create two propositional variables vA and vB for each validator v, where vA is true when v is in quorum A, and vB is true when v is in quorum B.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums and the truth assignment gives us the quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """

    quorum_tag: int = 1

    def in_quorum(q: str, n: str) -> Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return Atom((quorum_tag, q, n))

    def get_quorum_(atoms: list[int], q: str, fbas: FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [variables_inv[v][2] for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == quorum_tag and variables_inv[v][1] == q
                and variables_inv[v][2] in fbas.validators]

    start_time = time.time()
    constraints: list[Formula] = []
    for q in ['A', 'B']:  # our two quorums
        # the quorum must contain at least one validator for which we have a
        # qset:
        constraints += [Or(*[in_quorum(q, n)
                           for n in fbas.validators if fbas.threshold(n) >= 0])]
        constraints += quorum_constraints(fbas, lambda n: in_quorum(q, n))
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Or(Not(in_quorum('A', v)), Not(in_quorum('B', v)))]
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)
    start_time = time.time()
    clauses = to_cnf(constraints)
    end_time = time.time()
    logging.info("Time to convert to CNF: %s", end_time - start_time)

    res, s = solve_constraints(constraints)

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
        model = s.get_model() or []
        assert model  # error if []
        q1 = get_quorum_(model, 'A', fbas)
        q2 = get_quorum_(model, 'B', fbas)
        logging.info("Disjoint quorums found")
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1, over_approximate=True)
        assert fbas.is_quorum(q2, over_approximate=True)
        assert not set(q1) & set(q2)
        return (q1, q2)
    return None


def maximize(wcnf: WCNF) -> Optional[Tuple[int, Any]]:
    """
    Solve a MaxSAT CNF problem.
    """
    s: Union[LSU, RC2]
    if config.max_sat_algo == 'LSU':
        s = LSU(wcnf)
    else:
        s = RC2(wcnf)

    start_time = time.time()
    if isinstance(s, LSU):
        res = s.solve()
    else:  # RC2
        res = s.compute()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res:
        return s.cost, s.model
    return None


def find_minimal_splitting_set(
        fbas: FBASGraph) -> Optional[Tuple[Collection, Collection, Collection]]:
    """
    Find a minimal-cardinality splitting set in the FBAS graph, or prove there is none.
    Uses one of pysat's MaxSAT procedures (LSU or RC2).
    If found, returns the splitting set and the two quorums that it splits.
    """

    logging.info(
        "Finding minimal-cardinality splitting set using MaxSAT algorithm %s with %s cardinality encoding",
        config.max_sat_algo,
        config.card_encoding)

    faulty_tag: int = 0
    quorum_tag: int = 1

    def in_quorum(q: str, n: str) -> Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return Atom((quorum_tag, q, n))

    def get_quorum_(atoms: list[int], q: str, fbas: FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [variables_inv[v][2] for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == quorum_tag and variables_inv[v][1] == q
                and variables_inv[v][2] in fbas.validators]

    def faulty(n: str) -> Atom:
        """Returns an atom denoting whether node n is faulty."""
        return Atom((faulty_tag, n))

    def get_faulty(atoms: list[int]) -> list[str]:
        """Given a list of atoms, returns the faulty validators."""
        return [variables_inv[v][1]
                for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == faulty_tag]

    start_time = time.time()
    constraints: list[Formula] = []

    # now we create the constraints:
    for q in ['A', 'B']:  # for each of our two quorums
        # the quorum contains at least one non-faulty validator for which we
        # have a qset:
        constraints += [Or(*[And(in_quorum(q, n), Not(faulty(n)))
                           for n in fbas.validators if fbas.threshold(n) >= 0])]
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum(q, n) for n in fbas.graph.successors(v)]
                if v in fbas.validators:
                    # the threshold must be met only if the validator is not
                    # faulty:
                    constraints.append(
                        Implies(And(in_quorum(q, v), Not(faulty(v))), Card(fbas.threshold(v), *vs)))
                else:
                    # the threshold must be met:
                    constraints.append(
                        Implies(
                            in_quorum(
                                q, v), Card(
                                fbas.threshold(v), *vs)))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
    # add the constraint that no non-faulty validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Or(faulty(v), Not(in_quorum('A', v)),
                           Not(in_quorum('B', v)))]

    if config.group_by:
        # we add constraints assert that the group is faulty if and only if all
        # its members are faulty
        groups = set(
            fbas.vertice_attrs(v)[config.group_by]
            for v in fbas.validators)
        members = {g: [v for v in fbas.validators
                       if fbas.vertice_attrs(v)[config.group_by] == g]
                   for g in groups}
        for g in groups:
            constraints.append(
                Implies(faulty(g), And(*[faulty(v) for v in members[g]])))
            constraints.append(
                Implies(Or(*[faulty(v) for v in members[g]]), faulty(g)))

    # finally, convert to weighted CNF and add soft constraints that minimize
    # the number of faulty validators (or groups):
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    if not config.group_by:
        for v in fbas.validators:
            wcnf.append(to_cnf(Not(faulty(v)))[0], weight=1)
    else:
        for g in groups:  # type: ignore
            wcnf.append(to_cnf(Not(faulty(g)))[0], weight=1)

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    result = maximize(wcnf)

    if not result:
        logging.info("No splitting set found!")
        return None
    else:
        cost, model = result
        logging.info(
            "Found minimal-cardinality splitting set of size is %s:",
            cost)
        model = list(model)
        ss = get_faulty(model)
        if not config.group_by:
            logging.info("Minimal-cardinality splitting set: %s",
                         [fbas.with_name(s) for s in ss])
        else:
            logging.info("Minimal-cardinality splitting set (groups): %s",
                         [s for s in ss if s in groups])  # type: ignore
            logging.info("Minimal-cardinality splitting set (corresponding validators): %s",
                         [fbas.with_name(s) for s in ss if s not in groups])  # type: ignore
        q1 = get_quorum_(model, 'A', fbas)
        q2 = get_quorum_(model, 'B', fbas)
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
    Find a minimal-cardinality blocking set in the FBAS graph, or prove there is none.

    This is a bit more tricky than for splitting sets because we need to ensure that the
    "blocked-by" relation is well-founded (i.e. not circular). We achieve this by introducing a
    partial order on vertices and asserting that a vertex can only be blocked by vertices that are
    strictly lower in the order.
    """

    logging.info(
        "Finding minimal-cardinality blocking set using MaxSAT algorithm %s with %s cardinality encoding",
        config.max_sat_algo,
        config.card_encoding)

    start_time = time.time()

    if not fbas.validators:
        logging.info("No validators in the FBAS graph!")
        return None

    constraints: list[Formula] = []

    faulty_tag: int = 0
    blocked_tag: int = 1

    def faulty(v: str) -> Atom:
        return Atom((faulty_tag, v))

    def blocked(v: str) -> Atom:
        return Atom((blocked_tag, v))

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
            constraints.append(Or(faulty(v), blocked(v)))
        if v not in fbas.validators:
            constraints.append(Not(faulty(v)))
        if fbas.threshold(v) > 0:
            may_block = [And(Or(blocked(n), faulty(n)), lt(n, v))
                         for n in fbas.graph.successors(v)]
            constraints.append(
                Implies(
                    Card(
                        blocking_threshold(v),
                        *may_block),
                    blocked(v)))
            constraints.append(
                Implies(And(blocked(v), Not(faulty(v))),
                        Card(blocking_threshold(v), *may_block)))
        if fbas.threshold(v) == 0:
            constraints.append(Not(blocked(v)))

    # The lt relation must be a partial order (anti-symmetric and transitive).
    # For performance, lt only relates vertices that are in the same strongly
    # connected components (as otherwise there is no possible cycle anyway in
    # the blocking relation).
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
        # we add constraints assert that the group is faulty if and only if all
        # its members are faulty
        groups = set(fbas.vertice_attrs(v)[config.group_by]
                     for v in fbas.validators)
        members = {g: [v for v in fbas.validators
                       if fbas.vertice_attrs(v)[config.group_by] == g]
                   for g in groups}
        for g in groups:
            constraints.append(
                Implies(faulty(g), And(*[faulty(v) for v in members[g]])))
            constraints.append(
                Implies(Or(*[faulty(v) for v in members[g]]), faulty(g)))

    # convert to weighted CNF and add soft constraints that minimize the
    # number of faulty validators:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    if not config.group_by:
        for v in fbas.validators:
            wcnf.append(to_cnf(Not(faulty(v)))[0], weight=1)
    else:
        for g in groups:
            wcnf.append(to_cnf(Not(faulty(g)))[0], weight=1)

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    result = maximize(wcnf)

    if not result:
        logging.info("No blocking set found!")
        return None
    else:
        cost, model = result
        model = list(model)
        logging.info(
            "Found minimal-cardinality blocking set, size is %s",
            cost)
        s: list[str] = [variables_inv[v][1]
                        for v in set(model) & set(variables_inv.keys())
                        if variables_inv[v][0] == faulty_tag]
        if not config.group_by:
            logging.info("Minimal-cardinality blocking set: %s",
                         [fbas.with_name(v) for v in s])
        else:
            logging.info("Minimal-cardinality blocking set: %s",
                         [g for g in s if g in groups])
        vs = set(s) - groups
        assert fbas.closure(vs) == fbas.validators
        for vs2 in combinations(vs, cost-1):
            # TODO isn't this going to fail with groups?
            assert fbas.closure(vs2) != fbas.validators
        if not config.group_by:
            return s
        else:
            return [g for g in s if g in groups]


def min_history_loss_critical_set(
        fbas: FBASGraph) -> Tuple[Collection[str], Collection[str]]:
    """
    Return a set of minimal cardinality such that, should the validators in the set stop publishing valid history, the history may be lost.
    """

    logging.info(
        "Finding minimal-cardinality history-loss critical set using MaxSAT algorithm %s with %s cardinality encoding",
        config.max_sat_algo,
        config.card_encoding)

    constraints: list[Formula] = []

    in_critical_quorum_tag: int = 0
    hist_error_tag: int = 1
    in_crit_no_error_tag: int = 2

    def has_hist_error(v) -> Atom:
        return Atom((hist_error_tag, v))

    def in_critical_quorum(v) -> Atom:
        return Atom((in_critical_quorum_tag, v))

    def in_crit_no_error(v) -> Atom:
        return Atom((in_crit_no_error_tag, v))

    for v in fbas.validators:
        if fbas.vertice_attrs(v).get('historyArchiveHasError', True):
            constraints.append(has_hist_error(v))
        else:
            constraints.append(Not(has_hist_error(v)))

    for v in fbas.validators:
        constraints.append(
            Implies(
                in_critical_quorum(v), Not(
                    has_hist_error(v)), in_crit_no_error(v)))
        constraints.append(
            Implies(
                in_crit_no_error(v), And(
                    in_critical_quorum(v), Not(
                        has_hist_error(v)))))

    # the critical contains at least one validator for which we have a qset:
    constraints += [Or(*[in_critical_quorum(v)
                       for v in fbas.validators if fbas.threshold(v) >= 0])]
    constraints += quorum_constraints(fbas, in_critical_quorum)

    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # minimize the number of validators that are in the critical quorum but do
    # not have history errors:
    for v in fbas.validators:
        wcnf.append(to_cnf(Not(in_crit_no_error(v)))[0], weight=1)

    result = maximize(wcnf)

    if not result:
        raise ValueError("No critical set found! This should not happen.")
    else:
        cost, model = result
        logging.info(
            "Found minimal-cardinality history-critical set, size is %s",
            cost)
        model = list(model)
        min_critical = [variables_inv[v][1] for v in set(model) & set(
            variables_inv.keys()) if variables_inv[v][0] == in_crit_no_error_tag]
        quorum = [variables_inv[v][1]
                  for v in set(model) & set(variables_inv.keys())
                  if variables_inv[v][0] == in_critical_quorum_tag
                  and variables_inv[v][1] in fbas.validators]
        logging.info("Minimal-cardinality history-critical set: %s",
                     [fbas.with_name(v) for v in min_critical])
        logging.info("Quorum: %s", [fbas.with_name(v) for v in quorum])
        return (min_critical, quorum)


def find_min_quorum(
        fbas: FBASGraph,
        not_subset_of=None,
        not_equal_to=None,
        restrict_to_scc=True) -> Collection[str]:
    """
    Find a minimal quorum in the FBAS graph using pyqbf.
    If not_subset_of is a set of validators, then the quorum should contain at least one validator outside this set.
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

    quorum_tag: int = 1

    def in_quorum(q: str, n: str) -> Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return Atom((quorum_tag, q, n))

    def get_quorum_(atoms: list[int], q: str, fbas: FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [variables_inv[v][2] for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == quorum_tag and variables_inv[v][1] == q
                and variables_inv[v][2] in fbas.validators]

    def quorum_constraints_(q: str) -> list[Formula]:
        constraints: list[Formula] = []
        constraints += [Or(*[in_quorum(q, n)
                           for n in fbas.validators if fbas.threshold(n) > 0])]
        constraints += quorum_constraints(fbas, lambda n: in_quorum(q, n))
        return constraints

    # The set 'A' is a quorum in the scc:
    qa_constraints: list[Formula] = quorum_constraints_('A')

    # it contains at least one validator outside not_subset_of:
    if not_subset_of:
        qa_constraints += [Or(*[in_quorum('A', n)
                              for n in fbas.validators if n not in not_subset_of])]
    # it is not equal to any of the quorums in not_equal_to:
    if not_equal_to:
        for s in not_equal_to:
            assert set(s) <= fbas.validators
            qa_constraints += [Or(
                Or(*[Not(in_quorum('A', n)) for n in s]),
                Or(*[in_quorum('A', n) for n in fbas.validators - set(s)]))]

    # If 'B' is a subset of 'A', then 'B' is not a quorum:
    qb_quorum = And(*quorum_constraints_('B'))
    qb_subset_qa = And(
        *[Implies(in_quorum('B', n), in_quorum('A', n)) for n in fbas.validators])
    qb_constraints = Implies(qb_subset_qa, Not(qb_quorum))

    qa_clauses = to_cnf(qa_constraints)
    qb_clauses = to_cnf(qb_constraints)
    pcnf = PCNF(from_clauses=qa_clauses + qb_clauses)  # type: ignore

    qa_atoms: set[int] = atoms_of_clauses(qa_clauses)
    qb_vertex_atoms: set[int] = set(
        abs(variables[in_quorum('B', n).identifier]) for n in fbas.vertices())
    qb_tseitin_atoms: set[int] = \
        atoms_of_clauses(qb_clauses) - (qb_vertex_atoms | qa_atoms)

    pcnf.exists(
        *list(qa_atoms)).forall(
            *list(qb_vertex_atoms)).exists(*list(qb_tseitin_atoms))

    # solvers: 'depqbf', 'qute', 'rareqs', 'qfun', 'caqe'
    s = QSolver(name='depqbf', bootstrap_with=pcnf)  # type: ignore
    res = s.solve()
    if res:
        model = s.get_model()
        qa = get_quorum_(model, 'A', fbas)
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
    if not HAS_QBF:
        raise ImportError(
            "QBF support not available. Install with: pip install python-fbas[qbf]")

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
    Check if the overlay is FBA-resilient. That is, for every quorum Q, removing the complement of Q should not disconnect the overlay graph.
    """
    quorum_tag: int = 0

    def in_quorum(v: str) -> Atom:
        return Atom((quorum_tag, v))
    constraints: list[Formula] = []
    # the quorum is non-empty (contains a validator with a valid qset):
    constraints += [Or(*[in_quorum(v)
                       for v in fbas.validators if fbas.threshold(v) >= 0])]
    # the quorum constraints are satisfied:
    constraints += quorum_constraints(fbas, in_quorum)

    # now we assert the graph is disconnected
    reachable_tag: int = 1

    def reachable(v) -> Atom:
        return Atom((reachable_tag, v))
    # some node is reachable:
    constraints.append(Or(*[And(reachable(v), in_quorum(v))
                       for v in fbas.validators]))
    # nodes outside the quorum are not reachable:
    constraints += [Implies(reachable(v), in_quorum(v))
                    for v in fbas.validators]
    # for every two nodes in the quorum and with an edge between each other,
    # one is reachable iff the other is:
    constraints += [
        Implies(
            And(in_quorum(v1),
                in_quorum(v2)),
            equiv(reachable(v1),
                  reachable(v2)))
        for v1, v2 in overlay.edges() if v1 != v2]
    # some node in the quorum is unreachable:
    constraints.append(
        Or(*[And(Not(reachable(v)), in_quorum(v)) for v in fbas.validators]))

    res, solver = solve_constraints(constraints)
    if res:
        model = solver.get_model() or []
        assert model  # error if []
        q = [variables_inv[v][1] for v in set(model) & set(variables_inv.keys(
        )) if variables_inv[v][0] == quorum_tag and variables_inv[v][1] in fbas.validators]
        logging.info("Quorum %s is disconnected", q)
    else:
        logging.info("The overlay is FBA-resilient!")
    return not res


def num_not_blocked(fbas: FBASGraph, overlay: nx.Graph) -> int:
    """
    Returns the number of validators v such that v is not blocked by its set of neighbors in the overlay.

    If this returns 0 then we know that, for every quorum Q, if we remove Q from the overlay graph,
    then each remaining validator still has at least one neighbor. Note that this does not imply
    that the graph remains connected.
    """
    n = 0
    for v in fbas.validators:
        peers = list(overlay.neighbors(v)) if v in overlay else []
        if v not in fbas.closure(peers):
            n += 1
    return n


def is_fba_resilient_approx(fbas: FBASGraph, overlay: nx.Graph) -> bool:
    """
    Check if every node is blocked by its set of neighbors in the overlay. Note this does not guarantee connectivity under maximal failures (i.e. removing the complement of a quorum).
    """
    for v in fbas.validators:
        peers = list(overlay.neighbors(v)) if v in overlay else []
        if v not in fbas.closure(peers):
            return False
    return True
