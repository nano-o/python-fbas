"""
This module provides a simple implementation of propositional logic formulas, extended with
cardinality constraints. The goal is to avoid all the bookkeeping done by pysat, which makes things
too slow. The main functionality is conversion to CNF.
"""

from abc import ABC
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast
from pysat.card import CardEnc, EncType
import python_fbas.config as config

class Formula(ABC):
    """
    Abstract base class for propositional logic formulas.
    """

@dataclass
class Atom(Formula):
    """
    A propositional logic atom.
    """
    identifier: Any

@dataclass
class Not(Formula):
    """
    Negation.
    """
    operand: Formula

@dataclass(init=False)
class And(Formula):
    """
    Conjunction.
    """
    operands: list[Formula]

    def __init__(self, *operands: Formula): # do we really need to be able to write And(a,b,c) instead of And([a,b,c])?
        self.operands = list(operands)

@dataclass(init=False)
class Or(Formula):
    """
    Disjunction.
    """
    operands: list[Formula]

    def __init__(self, *operands: Formula):
        self.operands = list(operands)
    
@dataclass(init=False)
class Implies(Formula):
    """
    Implication. The last operand is the conclusion.
    """
    operands: list[Formula]

    def __init__(self, *operands: Formula):
        assert len(operands) >= 2
        self.operands = list(operands)

@dataclass(init=False)
class Card(Formula):
    """
    A cardinality constraint expressing that at least the treshold, out of the operands, are true.
    """
    threshold: int
    operands: list[Formula]

    def __init__(self, threshold: int, *operands: Formula):
        assert threshold > 0 and len(operands) >= threshold
        self.threshold = threshold
        self.operands = list(operands)

def equiv(f1, f2) -> Formula:
    return And(Implies(f1, f2), Implies(f2, f1))

# Now on to CNF conversion

Clauses = list[list[int]]

def atoms_of_clauses(cnf:Clauses) -> set[int]:
  """Returns the set of atoms appearing in a CNF formula."""
  return {abs(l) for clause in cnf for l in clause}

next_int:int = 1 # global variable to generate unique variable names
variables:dict[Any,int] = {} # maps variable names to integers
variables_inv:dict[int,Any] = {} # inverse of variables; maps integers to variable names

def var(v:Any) -> int:
    """
    Get the integer corresponding to a variable name.
    """
    global next_int
    if v not in variables:
        variables[v] = next_int
        variables_inv[next_int] = v
        next_int += 1
    return variables[v]

def anonymous_var() -> int:
    """
    Get the next integer variable; do not associate it with a name.
    """
    global next_int
    next_int += 1
    return next_int - 1

def to_cnf(arg: list[Formula]|Formula) -> Clauses:
    """
    Convert the formula to CNF. This is a very basic application of the Tseitin transformation. We
    are not expecting formulas to share subformulas, so we will not keep track of which variables
    correspond to which subformulas.
    
    By convention, the last clause in the CNF is a unit clause that is satisfied iff the formula is
    satisfied. This is unless we know that the formula is a the top-level.

    Note that this is a recursive function that will blow the stack if a formula is too deep (which
    we do not expect for our application).
    """

    def to_cnf_top(fmla: Formula) -> Clauses:
        """
        Only called at the top level (not recursively). Thus we do not need to ensure that last
        clause is a unit clause corresponding to the formula (which is normally used by the caller).
        """
        match fmla:
            case Or(ops):
                if all(isinstance(op, Atom) for op in ops):
                    return [[var(cast(Atom,a).identifier) for a in ops]]
                else:
                    return to_cnf(fmla)
            case Implies([And(ops), c]):
                if all(isinstance(op, Atom) for op in ops) and isinstance(c, Atom):
                    return [[-var(cast(Atom,a).identifier) for a in ops] + [var(c.identifier)]]
                else:
                    return to_cnf(fmla)
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
            return [c for f in fmlas for c in to_cnf_top(f)]
        case Atom() as a:
            return [[var(a.identifier)]]
        case Not(f):
            match f:
                case Atom(v):
                    return [[-var(v)]]
                case _:
                    v = anonymous_var()
                    op_clauses = to_cnf(f)
                    assert len(op_clauses[-1]) == 1
                    op_atom = op_clauses[-1][0] # that's the variable corresponding to the operand
                    new_clauses = [[-v, -op_atom],[v, op_atom]]
                    return op_clauses[:-1] + new_clauses + [[v]]
        case And(ops):
            if not ops:
                v = anonymous_var()
                return [[v]] # trivially satisfiable
            ops_clauses = [to_cnf(op) for op in ops]
            assert all(len(c[-1]) == 1 for c in ops_clauses)
            ops_atoms = [c[-1][0] for c in ops_clauses]
            inner_clauses = [c for cs in ops_clauses for c in cs[:-1]]
            return inner_clauses + and_gate(ops_atoms)
        case Or(ops):
            if not ops:
                v = anonymous_var()
                return [[-v],[v]] # unsatisfiable
            ops_clauses = [to_cnf(op) for op in ops]
            assert all(len(c[-1]) == 1 for c in ops_clauses)
            ops_atoms = [c[-1][0] for c in ops_clauses]
            inner_clauses = [c for cs in ops_clauses for c in cs[:-1]]
            return inner_clauses + or_gate(ops_atoms)
        case Implies(ops):
            return to_cnf(Or(Not(And(*ops[:-1])), ops[-1]))
        case Card(threshold, ops):
            match config.card_encoding:
                case 'naive':
                    # NOTE possibly lots of sub-formula sharing here: will be innefficient
                    fmla = Or(*[And(*c) for c in combinations(ops, threshold)])
                    return to_cnf(fmla)
                case 'totalizer':
                    ops_clauses = [to_cnf(op) for op in ops]
                    assert all(len(c[-1]) == 1 for c in ops_clauses)
                    ops_atoms = [c[-1][0] for c in ops_clauses]
                    inner_clauses = [c for cs in ops_clauses for c in cs[:-1]]
                    if threshold == len(ops_atoms):
                        return inner_clauses + and_gate(ops_atoms)
                    if threshold == 1:
                        return inner_clauses + or_gate(ops_atoms)
                    global next_int
                    cnfp = CardEnc.atleast(lits=list(ops_atoms), bound=threshold, top_id=next_int, encoding=EncType.totalizer)
                    next_int = cnfp.nv+1
                    return inner_clauses + cnfp.clauses
                case _:
                    raise ValueError('Unknown cardinality encoding')
        case _:
            raise TypeError
