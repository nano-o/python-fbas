import logging
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType


def test_1():
    lits = [1, 2, 3, 4]
    cnfp1 = CardEnc.atleast(
        bound=3,
        lits=lits,
        top_id=len(lits),
        encoding=EncType.totalizer)
    cnfp2 = CardEnc.atmost(
        bound=2,
        lits=lits,
        top_id=cnfp1.nv,
        encoding=EncType.totalizer)
    clauses = cnfp1.clauses[:-1]
    clauses += cnfp2.clauses[:-1]
    top_id = cnfp2.nv + 1
    clauses += [[-top_id, cnfp1.clauses[-1][0]],
                [top_id, cnfp2.clauses[-1][0]]]
    clauses += [[top_id]]  # set to negative to get atmost constraint
    logging.info("clauses: %s", clauses)
    s = Solver(bootstrap_with=clauses, name='cms')
    res = s.solve()
    assert res
    model = s.get_model()
    proj = [a for a in model if abs(a) in lits + [top_id]]
    logging.info("model: %s", proj)
