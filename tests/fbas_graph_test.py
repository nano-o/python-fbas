import logging
import pytest
import random
import json
from test_utils import get_test_data_list, load_fbas_from_test_file, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph
from python_fbas.serialization import deserialize
from python_fbas.config import temporary_config
from python_fbas.stellarbeat_serializer import qset_of


def test_load_fbas():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading fbas %s", f)
        # Use indulgent mode for test data that might have invalid entries
        with temporary_config(deserialization_mode='indulgent'):
            fg = deserialize(json.dumps(d))
        fg.check_integrity()


def test_is_quorum():
    fbas = load_fbas_from_test_file('conflicted.json')
    assert fbas.is_quorum({'PK11', 'PK12', 'PK13'})
    assert fbas.is_quorum({'PK11', 'PK12'})
    assert not fbas.is_quorum({'PK11'})
    assert not fbas.is_quorum({'PK11', 'PK12', 'PK13', 'PK21'})
    assert not fbas.is_quorum({'PK13', 'PK21'})
    assert fbas.is_quorum({'PK11', 'PK12', 'PK13', 'PK21', 'PK22', 'PK23'})
    with pytest.raises(AssertionError):
        fbas.is_quorum({'PK11', 'PK12', 'PK13', 'NON_EXISTENT'})
    assert not fbas.is_quorum({'PK11', 'PK12', 'PK13', 'PKX'})
    assert fbas.is_quorum({'PK11', 'PK12', 'PKX', 'PK22', 'PK23'})
    assert not fbas.is_quorum({'PK11', 'PK12', 'PKX', 'PK22'})


def test_is_quorum_2():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading fbas of %s", f)
        # Use indulgent mode for test data that might have invalid entries
        with temporary_config(deserialization_mode='indulgent'):
            fbas_graph = deserialize(json.dumps(d))
        if fbas_graph.get_validators():
            for _ in range(100):
                # pick a random subset of validators for which we have a qset:
                vs = [
                    v for v in fbas_graph.get_validators() if fbas_graph.has_qset(v)]
                n = random.randint(1, len(vs))
                validators = random.sample(vs, n)
                fbas_graph.is_quorum(validators)


def test_is_sat():
    fbas = load_fbas_from_test_file('circular_2.json')
    assert fbas.is_sat('PK3', {'PK3'})
    assert fbas.is_sat('PK2', {'PK3'})
    assert not fbas.is_sat('PK1', {'PK3'})


def test_find_disjoint_quorums():
    fbas1 = load_fbas_from_test_file('conflicted.json')
    q1, q2 = fbas1.find_disjoint_quorums()  # type: ignore
    logging.info("disjoint quorums: %s, %s", q1, q2)
    fbas2 = load_fbas_from_test_file('circular_1.json')
    assert not fbas2.find_disjoint_quorums()
    fbas3 = load_fbas_from_test_file('circular_2.json')
    assert not fbas3.find_disjoint_quorums()


def test_closure():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading fbas of %s", f)
        fbas_graph = deserialize(json.dumps(d))
        if fbas_graph.get_validators():
            for _ in range(100):
                # pick a random subset of validators:
                n = random.randint(1, len(fbas_graph.get_validators()))
                validators = random.sample(list(fbas_graph.get_validators()), n)
                assert fbas_graph.closure(validators)


def test_closure_2():
    fbas = load_fbas_from_test_file('circular_2.json')
    assert fbas.closure({'PK3'}) == {'PK1', 'PK2', 'PK3'}
    assert fbas.closure({'PK2'}) == {'PK1', 'PK2'}
    assert fbas.closure({'PK1'}) == {'PK1'}
    assert fbas.closure({'PK2', 'PK3'}) == {'PK1', 'PK2', 'PK3'}
    fbas2 = load_fbas_from_test_file('conflicted.json')
    assert fbas2.closure({'PK11', 'PK12'}) == {'PK11', 'PK12', 'PK13', 'PKX'}
    assert fbas2.closure({'PKX'}) == {'PKX'}
    assert fbas2.closure({'PK11', 'PK22'}) == {'PK11', 'PK22'}


def test_self_intersecing():
    fbas = FBASGraph()
    fbas.add_validator('PK1')
    fbas.add_validator('PK2')
    fbas.add_validator('PK3')

    # Test self-intersecting qset (threshold > half of members)
    n1 = fbas.add_qset(threshold=2, components=['PK1', 'PK2', 'PK3'], qset_id='n1')
    assert fbas.self_intersecting(n1)

    # Test non-self-intersecting qset (threshold <= half of members)
    n2 = fbas.add_qset(threshold=1, components=['PK1', 'PK2', 'PK3'], qset_id='n2')
    assert not fbas.self_intersecting(n2)

    # Validators are always self-intersecting
    assert fbas.self_intersecting('PK1')
    with pytest.raises(AssertionError):
        fbas.self_intersecting('XXX')


def test_intersection_bound():
    fbas = FBASGraph()

    # Add all validators first
    validators = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3',
                  'd1', 'd2', 'd3', 'e1', 'e2', 'e3', 'f1', 'f2', 'f3',
                  'g1', 'g2', 'g3', 'x', 'PK1', 'PK2', 'PK3']
    for v in validators:
        fbas.add_validator(v)

    # Create org qsets (inner quorum sets)
    org_a = fbas.add_qset(threshold=2, components=['a1', 'a2', 'a3'], qset_id='org_a')
    org_b = fbas.add_qset(threshold=2, components=['b1', 'b2', 'b3'], qset_id='org_b')
    org_c = fbas.add_qset(threshold=2, components=['c1', 'c2', 'c3'], qset_id='org_c')
    org_d = fbas.add_qset(threshold=2, components=['d1', 'd2', 'd3'], qset_id='org_d')
    org_e = fbas.add_qset(threshold=2, components=['e1', 'e2', 'e3'], qset_id='org_e')
    org_f = fbas.add_qset(threshold=2, components=['f1', 'f2', 'f3'], qset_id='org_f')
    org_g = fbas.add_qset(threshold=2, components=['g1', 'g2', 'g3'], qset_id='org_g')

    # Create main qsets with inner quorum sets
    n1 = fbas.add_qset(threshold=4, components=[org_a, org_b, org_c, org_d, org_e], qset_id='qset_1')
    n2 = fbas.add_qset(threshold=4, components=[org_b, org_c, org_d, org_e, org_f], qset_id='qset_2')
    n3 = fbas.add_qset(threshold=4, components=[org_c, org_d, org_e, org_f, org_g], qset_id='qset_3')
    n4 = fbas.add_qset(threshold=4, components=['x', org_c, org_d, org_e, org_f], qset_id='qset_4')

    assert fbas.intersection_bound_heuristic(n1, n2) == 2
    assert fbas.intersection_bound_heuristic(n1, n3) == 1
    assert fbas.intersection_bound_heuristic(n1, n4) == 1

    # Simple qsets for the second part
    nq1 = fbas.add_qset(threshold=2, components=['PK1', 'PK2', 'PK3'], qset_id='q1')
    nq2 = fbas.add_qset(threshold=2, components=['PK1', 'PK2', 'PK3'], qset_id='q2')
    nq3 = fbas.add_qset(threshold=1, components=['PK1', 'PK2', 'PK3'], qset_id='q3')

    assert fbas.intersection_bound_heuristic(nq1, nq2) == 1
    assert fbas.intersection_bound_heuristic(nq1, nq3) == 0


def test_fast_intersection_1():
    fbas = load_fbas_from_test_file('top_tier.json')
    assert fbas.fast_intersection_check() == 'true'
    fbas2 = load_fbas_from_test_file('validators.json')
    assert fbas2.fast_intersection_check() == 'true'


def test_fast_intersection_2():
    conflicted_fbas = load_fbas_from_test_file('conflicted.json')
    v11 = 'PK11'
    v23 = 'PK23'
    vx = 'PKX'
    assert conflicted_fbas.find_disjoint_quorums()  # there are disjoint quorums
    fbas1 = conflicted_fbas.restrict_to_reachable(v11)
    assert fbas1.fast_intersection_check() == 'true'
    fbas2 = conflicted_fbas.restrict_to_reachable(v23)
    assert fbas2.fast_intersection_check() == 'true'
    fbas3 = conflicted_fbas.restrict_to_reachable(vx)
    assert fbas3.fast_intersection_check() == 'unknown'


def test_fast_intersection_3():
    # This is an example where the fbas is intertwined but the fast heuristic
    # fails to see it
    circular_fbas = load_fbas_from_test_file('circular_1.json')
    assert not circular_fbas.find_disjoint_quorums()
    assert circular_fbas.fast_intersection_check() == 'unknown'


def test_fast_intersection_4():
    # This is an example where the fbas is intertwined but the fast heuristic
    # fails to see it
    circular_fbas = load_fbas_from_test_file('circular_2.json')
    assert not circular_fbas.find_disjoint_quorums()
    assert circular_fbas.fast_intersection_check() == 'unknown'


def test_is_qset_sat():
    fbas = FBASGraph()
    fbas.add_validator('PK1')
    fbas.add_validator('PK2')
    fbas.add_validator('PK3')
    fbas.add_validator('PK4')
    fbas.add_validator('PK5')

    # Simple qset with threshold 2 out of 3 validators
    n1 = fbas.add_qset(threshold=2, components=['PK1', 'PK2', 'PK3'], qset_id='n1')
    assert fbas.is_qset_sat(n1, {'PK1', 'PK2'})
    assert not fbas.is_qset_sat(n1, {'PK1'})

    # More complex qset with nested structure
    # First create inner qset
    inner_qset = fbas.add_qset(threshold=2, components=['PK1', 'PK2', 'PK3'], qset_id='inner_qset')
    # Then outer qset that needs 3 things: the inner qset + PK4 + PK5
    n2 = fbas.add_qset(threshold=3, components=['PK3', 'PK4', 'PK5', inner_qset], qset_id='n2')

    # This should NOT satisfy n2 (only satisfies inner_qset and PK3, but needs 3 total)
    assert not fbas.is_qset_sat(n2, {'PK1', 'PK2', 'PK3'})
    # This SHOULD satisfy n2 (inner_qset + PK3 + PK4 = 3 requirements met)
    assert fbas.is_qset_sat(n2, {'PK1', 'PK2', 'PK3', 'PK4'})
    # This should NOT satisfy n2 (inner_qset satisfied but only PK5 direct, missing one more)
    assert not fbas.is_qset_sat(n2, {'PK1', 'PK2', 'PK5'})


def test_qset_of():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading fbas %s", f)
        # Use indulgent mode for test data that might have invalid entries
        with temporary_config(deserialization_mode='indulgent'):
            fg = deserialize(json.dumps(d))
        for v in fg.get_validators():
            if list(fg.graph_view().successors(v)):
                qset_of(fg, v)


def test_format_validator():
    fbas = FBASGraph()
    fbas.add_validator("GABCD")
    fbas.update_validator("GABCD", qset=None, name='Test Validator')
    fbas.add_validator("GXYZ")  # Validator without a name

    with temporary_config(validator_display='id'):
        assert fbas.format_validator("GABCD") == "GABCD"
        assert fbas.format_validator("GXYZ") == "GXYZ"

    with temporary_config(validator_display='name'):
        assert fbas.format_validator("GABCD") == "Test Validator"
        assert fbas.format_validator("GXYZ") == "GXYZ"

    with temporary_config(validator_display='both'):
        assert fbas.format_validator("GABCD") == "GABCD (Test Validator)"
        assert fbas.format_validator("GXYZ") == "GXYZ"


def test_qset_reachability_check():
    """Test that qset vertices cannot reach themselves without passing through a validator vertex."""

    # Test case 1: Valid configuration - qset vertices only connected through validators
    fbas1 = FBASGraph()
    fbas1.add_validator('V1')
    fbas1.add_validator('V2')

    # Create qsets for validators - no direct qset-to-qset cycles
    qset_v1 = fbas1.add_qset(threshold=1, components=['V2'], qset_id='qset_v1')
    qset_v2 = fbas1.add_qset(threshold=1, components=['V1'], qset_id='qset_v2')

    # Connect validators to their qsets
    fbas1.update_validator('V1', qset=qset_v1)
    fbas1.update_validator('V2', qset=qset_v2)

    # This should pass - no qset cycles without validators
    fbas1.check_integrity()

    # Test case 2: Invalid configuration - qset vertices can reach themselves without validators
    fbas2 = FBASGraph()
    fbas2.add_validator('V1')
    fbas2.add_validator('V2')

    # Create two qsets that will form a cycle
    qset1 = fbas2.add_qset(threshold=1, components=['V1'], qset_id='qset1')
    qset2 = fbas2.add_qset(threshold=1, components=['V2'], qset_id='qset2')

    # Create a cycle between qsets: qset1 -> qset2 -> qset1
    # Remove existing edges to validators and add edges between qsets
    fbas2._graph.remove_edge(qset1, 'V1')
    fbas2._graph.remove_edge(qset2, 'V2')
    fbas2._graph.add_edge(qset1, qset2)
    fbas2._graph.add_edge(qset2, qset1)

    # This should fail the integrity check
    with pytest.raises(ValueError) as exc_info:
        fbas2.check_integrity()
    assert "cycle" in str(exc_info.value)

    # Test case 3: Complex valid case with multiple levels
    fbas3 = FBASGraph()
    fbas3.add_validator('V1')
    fbas3.add_validator('V2')
    fbas3.add_validator('V3')

    # Create nested structure: V1 -> inner_qset -> V3 and V2 -> outer_qset -> inner_qset
    inner_qset = fbas3.add_qset(threshold=1, components=['V3'], qset_id='inner_qset')
    outer_qset = fbas3.add_qset(threshold=2, components=['V2', 'inner_qset'], qset_id='outer_qset')

    # Connect validators to qsets
    fbas3.update_validator('V1', qset=inner_qset)
    fbas3.update_validator('V2', qset=outer_qset)

    # This should pass - all qset connections go through validators
    fbas3.check_integrity()
