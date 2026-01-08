from abc import ABC
import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast, Callable, Sequence
from pysat.card import ITotalizer
from python_fbas.config import get

"""
This module provides a simple implementation of propositional logic formulas,
extended with cardinality constraints. The goal is to avoid most of the
bookkeeping done by pysat, which makes things too slow, and use CNF
optimizations specific to our usecase. The main functionality is conversion to
equivalent CNF (up to existential quantification of auxiliary variables).

If card_encoding is set to 'totalizer', cardinality constraints use a totalizer
AtMost encoding on negated literals; negation of totalizer-encoded constraints
is not supported (see details in to_cnf).

To support negation of cardinality constraints, we could consider converting
formulas to NNF and rewriting Not(AtLeast(k, ops)) into AtMost(k-1, ops).
"""


# First we define classes to represent Boolean formulas.

class Formula(ABC):
    """
    Abstract base class for propositional logic formulas.
    """


@dataclass(slots=True)
class Atom(Formula):
    """
    A propositional logic atom.
    """
    identifier: Any


@dataclass(slots=True)
class Not(Formula):
    """
    Negation.
    """
    operand: Formula


@dataclass(init=False, slots=True)
class And(Formula):
    """
    Conjunction.
    """
    operands: list[Formula]

    def __init__(self, *operands: Formula): # do we really need to be able to write And(a,b,c) instead of And([a,b,c])?
        self.operands = list(operands)


@dataclass(init=False, slots=True)
class Or(Formula):
    """
    Disjunction.
    """
    operands: list[Formula]

    def __init__(self, *operands: Formula):
        self.operands = list(operands)


@dataclass(init=False, slots=True)
class Implies(Formula):
    """
    Implication. The last operand is the conclusion.
    """
    operands: list[Formula]

    def __init__(self, *operands: Formula):
        assert len(operands) >= 2
        self.operands = list(operands)


@dataclass(init=False, slots=True)
class AtLeast(Formula):
    """
    A cardinality constraint expressing that at least `threshold` number of
    operands are true.

    """
    threshold: int
    operands: list[Formula]

    def __init__(self, threshold: int, *operands: Formula):
        assert threshold > 0 and len(operands) >= threshold
        self.threshold = threshold
        self.operands = list(operands)


def equiv(f1, f2) -> Formula:
    return And(Implies(f1, f2), Implies(f2, f1))

def atoms_of_formula(fmla: Formula) -> set[Any]:
    """Returns the set of atoms appearing in a formula."""
    match fmla:
        case Atom(identifier):
            return {identifier}
        case Not(operand):
            return atoms_of_formula(operand)
        case And(operands) | Or(operands) | Implies(operands) | AtLeast(_, operands):
            return {atom for op in operands for atom in atoms_of_formula(op)}
        case _:
            raise TypeError


# Now on to CNF conversion.  Given a formula, the goal is to produce an
# equisatisfiable CNF formula over a new set of atoms and a bijection from the
# original atoms to the new atoms such that a model of the original formula can
# be extracted from a model of the CNF formula.

# A CNF formula is a list of clauses, where a clause is just a list of integers.
# An positive integer represents an atom. A negative integer represents the
# negation of the atom represented by the absolute value of the negative
# integer.
Clauses = list[list[int]]


def atoms_of_clauses(cnf: Clauses) -> set[int]:
    """Returns the set of atoms appearing in a CNF formula."""
    return {abs(literal) for clause in cnf for literal in clause}


# We use the global variable `next_int` to generate fresh CNF atoms. Note that
# this is never reset.
next_int: int = 1

# we use two dicts to keep track of which CNF atom represens which variable and
# vice versa:
# TODO: should this be marked private?
variables: dict[Any, int] = {}  # maps variable identifiers to CNF atoms
variables_inv: dict[int, Any] = {}  # inverse of variables; maps CNF atoms to variable identifiers


def var(v: Any) -> int:
    """
    Get the integer atom corresponding to a variable identifier, or reserve one if it
    does not exist.
    """
    global next_int
    if v not in variables:
        variables[v] = next_int
        variables_inv[next_int] = v
        next_int += 1
    return variables[v]


def anonymous_var() -> int:
    """
    Create a new integer atom; do not associate it with a variable identifier.
    """
    global next_int
    next_int += 1
    return next_int - 1


def to_cnf(arg: list[Formula] | Formula, polarity: int = 1) -> Clauses:
    """
    Recursive method to convert the formula (or list of clauses) to a CNF that
    is equivalent up to existential quantification of the auxiliary variables
    introduced.

    This is a very basic application of the Tseitin transformation. We are not
    expecting formulas to share subformulas, so we will not keep track of which
    variables correspond to which subformulas.

    By convention, the last clause in the CNF is a unit clause that is satisfied
    if and only if the formula is satisfied. Callers depend on this convention.
    This is unless we know that the formula is at the top-level and so there is
    no caller to make use of this.

    Cardinality constraints are a special case: the totalizer encoding enforces
    AtLeast(k) by adding a unit clause for AtMost(n-k) over negated literals.
    This is sufficient to enforce the constraint when the unit clause is kept,
    but it is not a full reification literal, so negating it is unsound. As a
    result, totalizer-encoded AtLeast constraints are rejected under negation.

    Note that this is a recursive function that will blow the stack if a formula
    is too deep (which we do not expect for our application).
    """


    def to_cnf_top(fmla: Formula) -> Clauses:
        """
        Only called at the top level (not recursively). Thus we do not need to
        ensure that the last clause is a unit clause corresponding to the
        formula (which is normally used by the caller), and use this to optimize
        some cases. We observe a large performance impact.

        TODO this "optimization" (3rd case) is causing issues in test_min_quorum_2; figure out why.
        """
        match fmla:
            case Or(ops) if all(isinstance(op, Atom) for op in ops):
                return [[var(cast(Atom, a).identifier) for a in ops]]
            case Implies([And(ops), c]) if (
                    all(isinstance(op, Atom) for op in ops) and isinstance(c, Atom)):
                return [[-var(cast(Atom, a).identifier) for a in ops]
                        + [var(c.identifier)]]
            case Implies(operands=atoms) if all(isinstance(a, Atom) for a in atoms):
                logging.info(f"to_cnf_top: {fmla}")
                # return to_cnf(Or(Not(And(*atoms[:-1])), atoms[-1]), polarity=1)
                return [[-var(cast(Atom, a).identifier) for a in atoms[:-1]]
                        + [var(cast(Atom, atoms[-1]).identifier)]]
            case _:
                return to_cnf(fmla)

    def and_gate(atoms: list[int]) -> Clauses:
        v = anonymous_var()
        clauses = [[-a for a in atoms] + [v]] + [[-v, a] for a in atoms]
        return clauses + [[v]]

    def or_gate(atoms: list[int]) -> Clauses:
        v = anonymous_var()
        clauses = [[-a, v] for a in atoms] + [[-v] + atoms]
        return clauses + [[v]]

    match arg:
        case list(fmlas):
            try:
                clauses = [c for f in fmlas for c in to_cnf_top(f)]
            except ValueError as e:
                raise RuntimeError(f"CNF conversion failed") from e
            return clauses
        case Atom() as a:
            return [[var(a.identifier)]]
        case Not(f):
            match f:
                case Atom(v):
                    return [[-var(v)]]
                case _:
                    v = anonymous_var()
                    op_clauses = to_cnf(f, -polarity)
                    assert len(op_clauses[-1]) == 1
                    op_atom = op_clauses[-1][0]  # that's the variable corresponding to the operand
                    new_clauses = [[-v, -op_atom], [v, op_atom]]
                    return op_clauses[:-1] + new_clauses + [[v]]
        case And(ops):
            if not ops:
                v = anonymous_var()
                return [[v]]  # trivially satisfiable
            if len(ops) == 1:
                return to_cnf(ops[0])
            ops_clauses = [to_cnf(op, polarity) for op in ops]
            assert all(len(c[-1]) == 1 for c in ops_clauses)
            ops_atoms = [c[-1][0] for c in ops_clauses]
            inner_clauses = [c for cs in ops_clauses for c in cs[:-1]]
            return inner_clauses + and_gate(ops_atoms)
        case Or(ops):
            if not ops:
                v = anonymous_var()
                return [[]]  # unsat
            if len(ops) == 1:
                return to_cnf(ops[0])
            ops_clauses = [to_cnf(op, polarity) for op in ops]
            assert all(len(c[-1]) == 1 for c in ops_clauses)
            ops_atoms = [c[-1][0] for c in ops_clauses]
            inner_clauses = [c for cs in ops_clauses for c in cs[:-1]]
            return inner_clauses + or_gate(ops_atoms)
        case Implies(ops):
            return to_cnf(Or(Not(And(*ops[:-1])), ops[-1]), polarity=polarity)
        case AtLeast(threshold, ops):
            encoding = get().card_encoding
            match encoding:
                case 'naive':
                    # NOTE possibly lots of sub-formula sharing here: will be innefficient
                    fmla = Or(*[And(*c) for c in combinations(ops, threshold)])
                    return to_cnf(fmla, polarity)
                case 'totalizer':
                    if polarity < 0:
                        # Consider converting to NNF and rewriting
                        # Not(AtLeast(k, ops)) as AtMost(k-1, ops) if needed.
                        raise ValueError('Totalizer encoding does not support negation')
                    ops_clauses = [to_cnf(op, polarity) for op in ops]
                    assert all(len(c[-1]) == 1 for c in ops_clauses)
                    ops_atoms = [c[-1][0] for c in ops_clauses]
                    inner_clauses = [c for cs in ops_clauses for c in cs[:-1]]
                    if threshold == len(ops_atoms):
                        return inner_clauses + and_gate(ops_atoms)
                    if threshold == 1:
                        return inner_clauses + or_gate(ops_atoms)

                    global next_int
                    neg_bound = len(ops_atoms) - threshold
                    tot = ITotalizer(
                        lits=[-lit for lit in ops_atoms],
                        ubound=neg_bound,
                        top_id=next_int)
                    atmost_lit = -tot.rhs[neg_bound]
                    tot_clauses = tot.cnf.clauses
                    next_int = tot.top_id + 1
                    tot.delete()
                    return inner_clauses + tot_clauses + [[atmost_lit]]
                case _:
                    raise ValueError('Unknown cardinality encoding')
        case _:
            raise TypeError


# Helpers

def decode_model(model: Sequence[int],
                 *,
                 predicate: Callable[[Any], bool] | None = None) -> list[Any]:
    """
    Translate a SAT model (list of integers) to corresponding the list of
    original variable that evaluate to true.

    The optional `predicate` paramter is used to filter the returned
    variable identifiers.
    """
    ids: list[Any] = []
    for lit in model:
        if lit < 0:
            continue
        ident = variables_inv.get(abs(lit))
        if ident is None:
            # Anonymous Tseitin variable – skip
            continue
        if predicate is None or predicate(ident):
            ids.append(ident)
    return ids


def simplify_cnf_with_literal(cnf: Clauses, literal: int) -> Clauses:
    """
    Simplify a CNF formula by assuming a literal is true.
    
    Args:
        cnf: The CNF formula (list of clauses, where each clause is a list of literals)
        literal: The literal to assume true. Positive means the atom is true,
                negative means the atom is false.
    
    Returns:
        Simplified CNF formula after unit propagation of the literal.
        
    The simplification rules are:
    - Remove all clauses containing the literal (they become true)
    - Remove the negation of the literal from all remaining clauses
    - Empty clauses indicate unsatisfiability
    """
    if literal == 0:
        raise ValueError("Literal cannot be 0")
    
    simplified = []
    neg_literal = -literal
    
    for clause in cnf:
        # If the clause contains the literal, it's satisfied, so skip it
        if literal in clause:
            continue
            
        # Remove the negation of the literal from the clause
        new_clause = [lit for lit in clause if lit != neg_literal]
        
        # Add the simplified clause (could be empty if it only contained neg_literal)
        simplified.append(new_clause)
    
    return simplified


def simplify_cnf_with_literals(cnf: Clauses, literals: Sequence[int]) -> Clauses:
    """
    Simplify a CNF formula by assuming multiple literals are true.
    
    Args:
        cnf: The CNF formula
        literals: Sequence of literals to assume true
        
    Returns:
        Simplified CNF formula after propagating all literals
    """
    result = cnf
    for literal in literals:
        result = simplify_cnf_with_literal(result, literal)
    return result
