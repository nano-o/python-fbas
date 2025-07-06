import pytest
import python_fbas.propositional_logic as pl
from python_fbas.propositional_logic import Atom, Not, And, Or, Card, to_cnf, decode_model
from python_fbas.solver import solve_sat
import python_fbas.config as cfg

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _reset():
    pl.variables.clear()
    pl.variables_inv.clear()
    pl.next_int = 1


def _solve(formula):
    cnf = to_cnf(formula)
    # Use a solver that is always shipped with python-sat
    res = solve_sat(cnf, name='g3', label="test-sat")
    return res.sat, decode_model(res.model)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestSatIntegration:
    def setup_method(self):
        _reset()

    def test_basic_sat(self):
        x, y = Atom('x'), Atom('y')
        sat, model = _solve(And(x, Or(y, Not(y))))
        assert sat
        assert 'x' in model                     # x must be true in every model

    def test_basic_unsat(self):
        x = Atom('x')
        sat, _ = _solve(And(x, Not(x)))
        assert not sat

    def test_cardinality_naive_sat(self, monkeypatch):
        monkeypatch.setattr(cfg, 'card_encoding', 'naive')
        a, b, c = Atom('a'), Atom('b'), Atom('c')
        sat, model = _solve(Card(2, a, b, c))
        assert sat
        assert len({'a', 'b', 'c'} & set(model)) >= 2   # at least 2 true

    def test_cardinality_totalizer_sat(self, monkeypatch):
        monkeypatch.setattr(cfg, 'card_encoding', 'totalizer')
        a, b, c = Atom('a'), Atom('b'), Atom('c')
        sat, _ = _solve(Card(2, a, b, c))
        assert sat

    def test_cardinality_conflict(self, monkeypatch):
        monkeypatch.setattr(cfg, 'card_encoding', 'totalizer')
        a, b, c = Atom('a'), Atom('b'), Atom('c')
        # require two true but force all three false → UNSAT
        sat, _ = _solve(And(Card(2, a, b, c), Not(a), Not(b), Not(c)))
        assert not sat
