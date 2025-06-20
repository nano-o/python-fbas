import logging
import pytest
import random
from test_utils import get_test_data_list, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph

def test_load_fbas():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading fbas %s", f)
        fg = FBASGraph.from_json(d)
        fg.check_integrity()

def test_collapse():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading fbas %s", f)
        fg = FBASGraph.from_json(d)
        fg.check_integrity()
        logging.info("graph of %s before flattening has %s nodes", f, fg.graph.number_of_nodes())
        fg.flatten_diamonds()
        logging.info("graph of %s after flattening has %s nodes", f, fg.graph.number_of_nodes())

def test_is_quorum():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert fbas.is_quorum({'PK11','PK12','PK13'})
    assert fbas.is_quorum({'PK11','PK12'})
    assert not fbas.is_quorum({'PK11'})
    assert not fbas.is_quorum({'PK11','PK12','PK13','PK21'})
    assert not fbas.is_quorum({'PK13','PK21'})
    assert fbas.is_quorum({'PK11','PK12','PK13','PK21','PK22','PK23'})
    with pytest.raises(AssertionError):
        fbas.is_quorum({'PK11','PK12','PK13','NON_EXISTENT'})
    assert not fbas.is_quorum({'PK11','PK12','PK13','PKX'})
    assert fbas.is_quorum({'PK11','PK12','PKX','PK22','PK23'})
    assert not fbas.is_quorum({'PK11','PK12','PKX','PK22'})

def test_is_quorum_2():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading fbas of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            for _ in range(100):
                # pick a random subset of validators for which we have a qset:
                vs = [v for v in fbas_graph.validators if fbas_graph.threshold(v) > 0]
                n = random.randint(1, len(vs))
                validators = random.sample(vs, n)
                fbas_graph.is_quorum(validators)

def test_is_sat():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert fbas.is_sat('PK3', {'PK3'})
    assert fbas.is_sat('PK2', {'PK3'})
    assert not fbas.is_sat('PK1', {'PK3'})

def test_find_disjoint_quorums():
    fbas1 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    q1, q2 = fbas1.find_disjoint_quorums() # type: ignore
    logging.info("disjoint quorums: %s, %s", q1, q2)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not fbas2.find_disjoint_quorums()
    fbas3 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not fbas3.find_disjoint_quorums()

def test_closure():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading fbas of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            for _ in range(100):
                # pick a random subset of validators:
                n = random.randint(1, len(fbas_graph.validators))
                validators = random.sample(list(fbas_graph.validators), n)
                assert fbas_graph.closure(validators)

def test_closure_2():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert fbas.closure({'PK3'}) == {'PK1', 'PK2', 'PK3'}
    assert fbas.closure({'PK2'}) == {'PK1', 'PK2'}
    assert fbas.closure({'PK1'}) == {'PK1'}
    assert fbas.closure({'PK2','PK3'}) == {'PK1', 'PK2', 'PK3'}
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert fbas2.closure({'PK11','PK12'}) == {'PK11','PK12','PK13','PKX'}
    assert fbas2.closure({'PKX'}) == {'PKX'}
    assert fbas2.closure({'PK11','PK22'}) == {'PK11','PK22'}

def test_self_intersecing():
    q1 = {'threshold' : 2, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    fbas = FBASGraph()
    n1 = fbas.add_qset(q1)
    assert fbas.self_intersecting(n1)
    q2 = {'threshold' : 1, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    n2 = fbas.add_qset(q2)
    assert not fbas.self_intersecting(n2)
    assert fbas.self_intersecting('PK1')
    with pytest.raises(AssertionError):
       fbas.self_intersecting('XXX')

def test_intersection_bound():
    org_a = {'threshold' : 2, 'validators' : ['a1','a2','a3'], 'innerQuorumSets' : []}
    org_b = {'threshold' : 2, 'validators' : ['b1','b2','b3'], 'innerQuorumSets' : []}
    org_c = {'threshold' : 2, 'validators' : ['c1','c2','c3'], 'innerQuorumSets' : []}
    org_d = {'threshold' : 2, 'validators' : ['d1','d2','d3'], 'innerQuorumSets' : []}
    org_e = {'threshold' : 2, 'validators' : ['e1','e2','e3'], 'innerQuorumSets' : []}
    org_f = {'threshold' : 2, 'validators' : ['f1','f2','f3'], 'innerQuorumSets' : []}
    org_g = {'threshold' : 2, 'validators' : ['g1','g2','g3'], 'innerQuorumSets' : []}
    qset_1 = {'threshold' : 4, 'validators' : [], 'innerQuorumSets' : [org_a, org_b, org_c, org_d, org_e]}
    qset_2 = {'threshold' : 4, 'validators' : [], 'innerQuorumSets' : [org_b, org_c, org_d, org_e, org_f]}
    qset_3 = {'threshold' : 4, 'validators' : [], 'innerQuorumSets' : [org_c, org_d, org_e, org_f, org_g]}
    qset_4 = {'threshold' : 4, 'validators' : ['x'], 'innerQuorumSets' : [org_c, org_d, org_e, org_f]}
    fbas = FBASGraph()
    n1 = fbas.add_qset(qset_1)
    n2 = fbas.add_qset(qset_2)
    n3 = fbas.add_qset(qset_3)
    n4 = fbas.add_qset(qset_4)
    assert fbas.intersection_bound_heuristic(n1, n2) == 2
    assert fbas.intersection_bound_heuristic(n1, n3) == 1
    assert fbas.intersection_bound_heuristic(n1, n4) == 1    
    q1 = {'threshold' : 2, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    nq1 = fbas.add_qset(q1)
    q2 = {'threshold' : 2, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    nq2 = fbas.add_qset(q2)
    q3 = {'threshold' : 1, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    nq3 = fbas.add_qset(q3)
    assert fbas.intersection_bound_heuristic(nq1, nq2) == 1
    assert fbas.intersection_bound_heuristic(nq1, nq3) == 0

def test_fast_intersection_1():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('top_tier.json'))
    assert fbas.fast_intersection_check() == 'true'
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('validators.json'))
    assert fbas2.fast_intersection_check() == 'true'

def test_fast_intersection_2():
    conflicted_fbas = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    v11 = 'PK11'
    v23 = 'PK23'
    vx = 'PKX'
    assert conflicted_fbas.find_disjoint_quorums() # there are disjoint quorums
    fbas1 = conflicted_fbas.restrict_to_reachable(v11)
    assert fbas1.fast_intersection_check() == 'true'
    fbas2 = conflicted_fbas.restrict_to_reachable(v23)
    assert fbas2.fast_intersection_check() == 'true'
    fbas3 = conflicted_fbas.restrict_to_reachable(vx)
    assert fbas3.fast_intersection_check() == 'unknown'

def test_fast_intersection_3():
    # This is an example where the fbas is intertwined but the fast heuristic fails to see it
    circular_fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not circular_fbas.find_disjoint_quorums()
    assert circular_fbas.fast_intersection_check() == 'unknown'

def test_fast_intersection_4():
    # This is an example where the fbas is intertwined but the fast heuristic fails to see it
    circular_fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not circular_fbas.find_disjoint_quorums()
    assert circular_fbas.fast_intersection_check() == 'unknown'

def test_is_qset_sat():
    q1 = {'threshold' : 2, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    fbas = FBASGraph()
    n1 = fbas.add_qset(q1)
    assert fbas.is_qset_sat(n1, {'PK1','PK2'})
    assert not fbas.is_qset_sat(n1, {'PK1'})
    q2 = {'threshold' : 3, 'validators' : ['PK3','PK4','PK5'], 'innerQuorumSets' : [
        {'threshold' : 2, 'validators' : ['PK1','PK2','PK3'], 'innerQuorumSets' : []}
    ]}
    n2 = fbas.add_qset(q2)
    assert not fbas.is_qset_sat(n2, {'PK1','PK2','PK3'})
    assert fbas.is_qset_sat(n2, {'PK1','PK2','PK3','PK4'})
    assert not fbas.is_qset_sat(n2, {'PK1','PK2','PK5'})

def test_qset_of():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading fbas %s", f)
        fg = FBASGraph.from_json(d)
        for v in fg.validators:
            if list(fg.graph.successors(v)):
                qset = fg.qset_of(v)
                qset_vertex = fg.qset_vertex_of(v)
                assert qset == fg.qsets[qset_vertex], f"QSet of {v} does not match the expected QSet: {fg.qsets[qset_vertex]} != {qset}"