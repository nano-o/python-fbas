"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Any, Optional, Tuple, Collection
from itertools import combinations
import networkx as nx
from pysat.solvers import Solver
from pysat.formula import CNF, WCNF
from pysat.examples.lsu import LSU # MaxSAT algorithm
from pysat.examples.rc2 import RC2 # MaxSAT algorithm
from pyqbf.formula import PCNF
from pyqbf.solvers import Solver as QSolver
from python_fbas.fbas_graph import FBASGraph
from python_fbas.propositional_logic import And, Or, Implies, Atom, Formula, Card, Not, variables, variables_inv, to_cnf, Clauses
import python_fbas.config as config

def contains_quorum(s:set[str], fbas: FBASGraph) -> bool:
    """
    Check if s is a quorum in the FBAS graph.
    """
    assert s <= fbas.validators
    constraints:list[Formula] = []
    # the quorum must contain at least one validator from s (and for which we have a qset):
    constraints += [Or(*[Atom(v) for v in s if fbas.threshold(v) >= 0])]
    # no validators outside s are in the quorum:
    constraints += [And(*[Not(Atom(v)) for v in fbas.validators if v not in s])]
    # then, we add the threshold constraints (TODO factor this out):
    for v in fbas.vertices():
        if fbas.threshold(v) > 0:
            vs = [Atom(n) for n in fbas.graph.successors(v)]
            constraints.append(Implies(Atom(v), Card(fbas.threshold(v), *vs)))
        if fbas.threshold(v) == 0:
            continue # no constraints for this vertex
        if fbas.threshold(v) < 0:
            # to be conservative (i.e. create as many quorums as possible), no constraints
            continue

    # TODO factor out calling the solver and printing runtime
    clauses = to_cnf(constraints)
    solver = Solver(bootstrap_with=clauses, name=config.sat_solver)
    start_time = time.time()
    res = solver.solve()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res:
        model = solver.get_model()
        q = [variables_inv[v] for v in set(model) & set(variables_inv.keys()) if variables_inv[v] in fbas.validators]
        logging.info("Quorum %s is inside %s", q, s)
    else:
        logging.info("No quorum found in %s", s)
    return res

def find_disjoint_quorums(fbas: FBASGraph) -> Optional[Tuple[Collection, Collection]]:
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

    quorum_tag:int = 1

    def in_quorum(q:str, n:str) -> Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return Atom((quorum_tag, q, n))

    def get_quorum_(atoms:list[int], q:str, fbas:FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [variables_inv[v][2] for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == quorum_tag and variables_inv[v][1] == q
                    and variables_inv[v][2] in fbas.validators]
    
    start_time = time.time()
    constraints : list[Formula] = []
    for q in ['A', 'B']: # our two quorums
        # the quorum must contain at least one validator for which we have a qset:
        constraints += [Or(*[in_quorum(q, n) for n in fbas.validators if fbas.threshold(n) >= 0])]
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum(q, n) for n in fbas.graph.successors(v)]
                constraints.append(Implies(in_quorum(q, v), Card(fbas.threshold(v), *vs)))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: 
                # to be conservative (i.e. create as many quorums as possible), no constraints
                continue
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Or(Not(in_quorum('A', v)), Not(in_quorum('B', v)))]
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)
    start_time = time.time()
    clauses = to_cnf(constraints)
    end_time = time.time()
    logging.info("Time to convert to CNF: %s", end_time - start_time)

    # now call the solver:
    s = Solver(bootstrap_with=clauses, name=config.sat_solver)
    start_time = time.time()
    res = s.solve()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)

    if config.output:
        if config.output:
            with open(config.output, 'w', encoding='utf-8') as f:
                cnf = CNF(from_clauses=clauses)
                dimacs = cnf.to_dimacs()
                comment = "c " + ("SATISFIABLE" if res else "UNSATISFIABLE") + "\n"
                f.write(comment)
                f.writelines(dimacs)

    if res:
        model = s.get_model()
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
    
def maximize(wcnf:WCNF) -> Optional[Tuple[int, Any]]:
    """
    Solve a MaxSAT CNF problem.
    """
    if config.max_sat_algo == 'LSU':
        s = LSU(wcnf)
    else:
        s = RC2(wcnf)
    start_time = time.time()
    if config.max_sat_algo == 'LSU':
        res = s.solve()
    else:
        res = s.compute()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res:
        return s.cost, s.model
    return None

def find_minimal_splitting_set(fbas: FBASGraph) ->  Optional[Tuple[Collection,Collection,Collection]]:
    """
    Find a minimal-cardinality splitting set in the FBAS graph, or prove there is none.
    Uses one of pysat's MaxSAT procedures (LRU or RC2).
    If found, returns the splitting set and the two quorums that it splits.
    """

    logging.info("Finding minimal-cardinality splitting set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    faulty_tag:int = 0
    quorum_tag:int = 1

    def in_quorum(q:str, n:str) -> Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return Atom((quorum_tag, q, n))

    def get_quorum_(atoms:list[int], q:str, fbas:FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [variables_inv[v][2] for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == quorum_tag and variables_inv[v][1] == q
                    and variables_inv[v][2] in fbas.validators]

    def faulty(n:str) -> Atom:
        """Returns an atom denoting whether node n is faulty."""
        return Atom((faulty_tag,n))

    def get_faulty(atoms:list[int]) -> list[str]:
        """Given a list of atoms, returns the faulty validators."""
        return [variables_inv[v][1] for v in set(atoms) & set(variables_inv.keys()) \
                if variables_inv[v][0] == faulty_tag]
    
    start_time = time.time()
    constraints : list[Formula] = []

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # the quorum contains at least one non-faulty validator for which we have a qset:
        constraints += [Or(*[And(in_quorum(q, n), Not(faulty(n))) for n in fbas.validators if fbas.threshold(n) >= 0])]
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum(q, n) for n in fbas.graph.successors(v)]
                if v in fbas.validators:
                    # the threshold must be met only if the validator is not faulty:
                    constraints.append(Implies(And(in_quorum(q, v), Not(faulty(v))), Card(fbas.threshold(v), *vs)))
                else:
                    # the threshold must be met:
                    constraints.append(Implies(in_quorum(q, v), Card(fbas.threshold(v), *vs)))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0:
                # to be conservative (i.e. create as many quorums as possible), no constraints
                continue
    # add the constraint that no non-faulty validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Or(faulty(v), Not(in_quorum('A', v)), Not(in_quorum('B', v)))]

    if config.group_by:
        # we add constraints assert that the group is faulty if and only if all its members are faulty
        groups = set(fbas.vertice_attrs(v)[config.group_by] for v in fbas.validators)
        members = {g: [v for v in fbas.validators if fbas.vertice_attrs(v)[config.group_by] == g] for g in groups}
        for g in groups:
            constraints.append(Implies(faulty(g), And(*[faulty(v) for v in members[g]])))
            constraints.append(Implies(Or(*[faulty(v) for v in members[g]]), faulty(g)))

    # finally, convert to weighted CNF and add soft constraints that minimize the number of faulty validators (or groups):
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
        logging.info("No splitting set found!")
        return None
    else:
        cost, model = result
        logging.info("Found minimal-cardinality splitting set, size is %s", cost)
        model = list(model)
        ss = get_faulty(model)
        if not config.group_by:
            logging.info("Minimal-cardinality splitting set: %s", [fbas.with_name(s) for s in ss])
        else:
            logging.info("Minimal-cardinality splitting set (groups): %s", [s for s in ss if s in groups])
            logging.info("Minimal-cardinality splitting set (corresponding validators): %s", [fbas.with_name(s) for s in ss if s not in groups])
        q1 = get_quorum_(model, 'A', fbas)
        q2 = get_quorum_(model, 'B', fbas)
        logging.info("Quorum A: %s", [fbas.with_name(v) for v in q1])
        logging.info("Quorum B: %s", [fbas.with_name(v) for v in q2])
        if not config.group_by:
            return (ss, q1, q2)
        else:
            return ([s for s in ss if s in groups], q1, q2)

def find_minimal_blocking_set(fbas: FBASGraph) -> Optional[Collection[str]]:
    """
    Find a minimal-cardinality blocking set in the FBAS graph, or prove there is none.

    This is a bit more tricky than for splitting sets because we need to ensure that the
    "blocked-by" relation is well-founded (i.e. not circular). We achieve this by introducing a
    partial order on vertices and asserting that a vertex can only be blocked by vertices that are
    strictly lower in the order.
    """

    logging.info("Finding minimal-cardinality blocking set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    start_time = time.time()

    if not fbas.validators:
        logging.info("No validators in the FBAS graph!")
        return None

    constraints : list[Formula] = []

    faulty_tag:int = 0
    blocked_tag:int = 1

    def faulty(v:str) -> Atom:
        return Atom((faulty_tag,v))
    
    def blocked(v:str) -> Atom:
        return Atom((blocked_tag,v))

    def lt(v1:str, v2:str) -> Formula:
        """
        v1 is strictly lower than v2
        """
        return Atom((v1, v2))
    
    def blocking_threshold(v) -> int:
        return len(list(fbas.graph.successors(v))) - fbas.threshold(v) + 1
    
    # first, the threshold constraints:
    for v in fbas.vertices():
        constraints.append(Or(faulty(v), blocked(v)))
        if v not in fbas.validators:
            constraints.append(Not(faulty(v)))
        if fbas.threshold(v) > 0:
            may_block = [And(blocked(n), lt(n,v)) for n in fbas.graph.successors(v)]
            constraints.append(Implies(Card(blocking_threshold(v), *may_block), blocked(v)))
            constraints.append(Implies(And(blocked(v), Not(faulty(v))), Card(blocking_threshold(v), *may_block)))
        if fbas.threshold(v) == 0:
            # never blocked
            constraints.append(Not(blocked(v)))
        if fbas.threshold(v) < 0:
            # to be conservative, could be blocked by anything; so, no constraints
            continue

    # The lt relation must be a partial order (anti-symmetric and transitive). For performance, lt
    # only relates vertices that are in the same strongly connected components (as otherwise there
    # is no possible cycle in the blocking relation).
    sccs = [scc for scc in nx.strongly_connected_components(fbas.graph)
            if any(fbas.threshold(v) >= 0 for v in set(scc))]
    assert sccs
    for scc in sccs:
        for v1 in scc:
            constraints.append(Not(lt(v1, v1)))
            for v2 in scc:
                for v3 in scc:
                    constraints.append(Implies(And(lt(v1, v2), lt(v2, v3)), lt(v1, v3)))
    
    groups = set()
    if config.group_by:
        # we add constraints assert that the group is faulty if and only if all its members are faulty
        groups = set(fbas.vertice_attrs(v)[config.group_by] for v in fbas.validators)
        members = {g: [v for v in fbas.validators if fbas.vertice_attrs(v)[config.group_by] == g] for g in groups}
        for g in groups:
            constraints.append(Implies(faulty(g), And(*[faulty(v) for v in members[g]])))
            constraints.append(Implies(Or(*[faulty(v) for v in members[g]]), faulty(g)))

    # convert to weighted CNF and add soft constraints that minimize the number of faulty validators:
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
        logging.info("Found minimal-cardinality blocking set, size is %s", cost)
        s:list[str] = [variables_inv[v][1] for v in set(model) & set(variables_inv.keys()) \
            if variables_inv[v][0] == faulty_tag]
        if not config.group_by:
            logging.info("Minimal-cardinality blocking set: %s", [fbas.with_name(v) for v in s])
        else:
            logging.info("Minimal-cardinality blocking set: %s", [g for g in s if g in groups])
        vs = [v for v in s if v not in groups]
        no_qset = {v for v in fbas.validators if fbas.threshold(v) < 0}
        all_ = set(vs) | no_qset
        assert fbas.closure(all_) == fbas.validators
        for vs2 in combinations(all_, cost-1+len(no_qset)):
            assert fbas.closure(vs2) != fbas.validators
        if not config.group_by:
            return s
        else:
            return [g for g in s if g in groups]
        
def min_history_loss_critical_set(fbas: FBASGraph) -> Tuple[Collection[str], Collection[str]]:
    """
    Return a set of minimal cardinality such that, should the validators in the set stop publishing valid history, the history may be lost.
    """

    logging.info("Finding minimal-cardinality history-loss critical set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    constraints : list[Formula] = []

    in_critical_quorum_tag:int = 0
    hist_error_tag:int = 1
    in_crit_no_error_tag:int = 2

    def has_hist_error(v) -> Atom:
        return Atom((hist_error_tag,v))
    
    def in_critical_quorum(v) -> Atom:
        return Atom((in_critical_quorum_tag,v))
    
    def in_crit_no_error(v) -> Atom:
        return Atom((in_crit_no_error_tag,v))

    for v in fbas.validators:
        if fbas.vertice_attrs(v).get('historyArchiveHasError', True):
            constraints.append(has_hist_error(v))
        else:
            constraints.append(Not(has_hist_error(v)))

    for v in fbas.validators:
        constraints.append(Implies(in_critical_quorum(v), Not(has_hist_error(v)), in_crit_no_error(v)))
        constraints.append(Implies(in_crit_no_error(v), And(in_critical_quorum(v), Not(has_hist_error(v)))))

    # the critical contains at least one validator for which we have a qset:
    constraints += [Or(*[in_critical_quorum(v) for v in fbas.validators if fbas.threshold(v) >= 0])]
    for v in fbas.vertices():
        if fbas.threshold(v) > 0:
            vs = [in_critical_quorum(n) for n in fbas.graph.successors(v)]
            constraints.append(Implies(in_critical_quorum(v), Card(fbas.threshold(v), *vs)))
        if fbas.threshold(v) == 0:
            continue # no constraints for this vertex
        if fbas.threshold(v) < 0:
            constraints.append(Not(in_critical_quorum(v)))

    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # minimize the number of validators that are in the critical quorum but do not have history errors:
    for v in fbas.validators:
        wcnf.append(to_cnf(Not(in_crit_no_error(v)))[0], weight=1)

    result = maximize(wcnf)

    if not result:
        raise ValueError("No critical set found! This should not happen.")
    else:
        cost, model = result
        logging.info("Found minimal-cardinality history-critical set, size is %s", cost)
        model = list(model)
        min_critical = [variables_inv[v][1] for v in set(model) & set(variables_inv.keys()) \
            if variables_inv[v][0] == in_crit_no_error_tag]
        quorum = [variables_inv[v][1] for v in set(model) & set(variables_inv.keys()) \
            if variables_inv[v][0] == in_critical_quorum_tag and variables_inv[v][1] in fbas.validators]
        logging.info("Minimal-cardinality history-critical set: %s", [fbas.with_name(v) for v in min_critical])
        logging.info("Quorum: %s", [fbas.with_name(v) for v in quorum])
        return (min_critical, quorum)
    
def find_min_quorum(fbas: FBASGraph) -> Collection[str]:
    """
    Find a minimal quorum in the FBAS graph using pyqbf.
    """

    # TODO: look only in sccs that contain a quorum

    if not fbas.validators:
        logging.info("The FBAS is empty!")
        return []

    quorum_tag:int = 1

    def in_quorum(q:str, n:str) -> Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return Atom((quorum_tag, q, n))
    
    def get_quorum_(atoms:list[int], q:str, fbas:FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [variables_inv[v][2] for v in set(atoms) & set(variables_inv.keys())
                if variables_inv[v][0] == quorum_tag and variables_inv[v][1] == q
                    and variables_inv[v][2] in fbas.validators]
    
    def quorum_constraints(q:str) -> list[Formula]:
        constraints:list[Formula] = []
        constraints += [Or(*[in_quorum(q, n) for n in fbas.validators if fbas.threshold(n) >= 0])]
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum(q, n) for n in fbas.graph.successors(v)]
                constraints.append(Implies(in_quorum(q, v), Card(fbas.threshold(v), *vs)))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: 
                # to be conservative (i.e. create as many quorums as possible), no constraints
                continue
        return constraints

    def atoms(cnf:Clauses) -> set[int]:
        return set([abs(l) for clause in cnf for l in clause])

    qa_constraints:list[Formula] = quorum_constraints('A')

    qb_quorum = And(*quorum_constraints('B'))
    qb_subset_qa = And(*([Or(Not(in_quorum('B', n)), in_quorum('A', n)) for n in fbas.validators] + [Or(And(in_quorum('A', n), Not(in_quorum('B', n)))) for n in fbas.validators]))
    qb_constraints = Implies(qb_subset_qa, Not(qb_quorum))

    qa_clauses = to_cnf(qa_constraints)
    qb_clauses = to_cnf(qb_constraints)
    pcnf = PCNF(from_clauses=qa_clauses + qb_clauses)

    qa_atoms:set[int] = atoms(qa_clauses)
    qb_vertex_atoms:set[int] = set(abs(variables[in_quorum('B', n).identifier]) for n in fbas.vertices())
    qb_tseitin_atoms:set[int] = atoms(qb_clauses) - (qb_vertex_atoms | qa_atoms)

    pcnf.exists(*list(qa_atoms)).forall(*list(qb_vertex_atoms)).exists(*list(qb_tseitin_atoms))
    
    # solvers: 'depqbf', 'qute', 'rareqs', 'qfun', 'caqe'
    s = QSolver(name='depqbf', bootstrap_with=pcnf)
    res = s.solve()
    if res:
        model = s.get_model()
        qa = get_quorum_(model, 'A', fbas)
        logging.info("Minimal quorum found: %s", qa)
        return qa
    else:
        logging.info("No minimal quorum found!")
        return []