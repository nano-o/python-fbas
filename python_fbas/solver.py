"""
Unified wrappers around SAT, MaxSAT and QBF back-ends.

These helpers centralise timing, configuration defaults and result handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from pysat.formula import CNF, WCNF
from pysat.solvers import Solver, SolverNames
from pysat.examples.lsu import LSU
from pysat.examples.rc2 import RC2

from python_fbas.utils import timed
from python_fbas.config import get

try:
    from pyqbf.formula import PCNF
    from pyqbf.solvers import Solver as QSolver  # type: ignore
    _HAS_QBF = True
except ImportError:
    _HAS_QBF = False

solvers: list[str] = [list(SolverNames.__dict__[s])[::-1][0]
                      for s in SolverNames.__dict__ if not s.startswith('__')]

# Export whether QBF support is present so callers/tests can inspect it
HAS_QBF: bool = _HAS_QBF

# ---------------------------------------------------------------------------#
# Result containers
# ---------------------------------------------------------------------------#
@dataclass
class SatResult:
    sat: bool
    model: List[int]


@dataclass
class MaxSatResult:
    sat: bool
    optimum: Optional[int]
    model: List[int]


@dataclass
class QbfResult:
    sat: bool
    model: List[int]


# ---------------------------------------------------------------------------#
# Solver helpers
# ---------------------------------------------------------------------------#
def solve_sat(clauses: Sequence[Sequence[int]],
              *,
              name: str | None = None,
              dimacs_out: Optional[str] = None,
              label: str = "SAT solving") -> SatResult:
    """
    Solve a plain CNF `clauses` and return the satisfiability status together
    with a model (empty when UNSAT).
    """
    if dimacs_out:
        CNF(from_clauses=clauses).to_file(dimacs_out)

    if name is None:
        name = get().sat_solver
    with timed(label):
        s = Solver(name=name, bootstrap_with=clauses)
        sat = bool(s.solve())
        model = s.get_model() or []

    return SatResult(sat, model)


def solve_maxsat(wcnf: WCNF,
                 *,
                 algo: str | None = None,
                 label: str = "MaxSAT solving") -> MaxSatResult:
    """
    Solve a weighted CNF instance using LSU or RC2 according to `algo`.
    """
    if algo is None:
        algo = get().max_sat_algo
    assert algo in ('LSU', 'RC2')
    engine: Union[LSU, RC2] = LSU(wcnf) if algo == 'LSU' else RC2(wcnf)

    with timed(label):
        sat = bool(engine.solve() if isinstance(engine, LSU) else engine.compute())

    return MaxSatResult(sat,
                        engine.cost if sat else None,
                        list(engine.model) if sat else [])


# solvers: 'depqbf', 'qute', 'rareqs', 'qfun', 'caqe'
def solve_qbf(pcnf: "PCNF",
              *,
              name: str = "depqbf",
              label: str = "QBF solving") -> QbfResult:
    """
    Solve a prenex CNF using pyqbf. Requires the optional 'qbf' extra.
    """
    if not _HAS_QBF:
        raise ImportError("QBF support not available. Install with: pip install python-fbas[qbf]")

    with timed(label):
        s = QSolver(name=name, bootstrap_with=pcnf)  # type: ignore
        sat = bool(s.solve())
        model = s.get_model() or []

    return QbfResult(sat, model)
