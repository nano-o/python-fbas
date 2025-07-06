import logging
import networkx as nx
from test_utils import get_test_data_list, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph
from python_fbas import config
from python_fbas.fbas_graph_analysis import (
    find_disjoint_quorums,
    find_minimal_splitting_set,
    find_minimal_blocking_set,
    find_min_quorum,
    contains_quorum,
    top_tier,
    is_overlay_resilient,
    num_not_blocked,
)
from python_fbas.solver import HAS_QBF


def test_qi_():
    config.card_encoding = 'totalizer'
    fbas = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums(fbas)
    fbas = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums(fbas)
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums(fbas2)
    config.card_encoding = 'naive'
    fbas = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums(fbas)
    fbas = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums(fbas)
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums(fbas2)


def test_qi_missing():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('missing_1.json'))
    assert not find_disjoint_quorums(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('missing_2.json'))
    assert find_disjoint_quorums(fbas)


def test_qi_all():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            config.card_encoding = 'totalizer'
            find_disjoint_quorums(fbas_graph)
            config.card_encoding = 'naive'
            find_disjoint_quorums(fbas_graph)


def test_min_splitting_set_1():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    assert len(find_minimal_splitting_set(fbas1)[0]) == 2  # type: ignore
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_1.json'))
    assert not find_minimal_splitting_set(fbas2)
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_2.json'))
    assert find_minimal_splitting_set(fbas2)[0] == ['PK2']  # type: ignore


def test_min_splitting_set_2():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    assert len(find_minimal_splitting_set(fbas1)[0]) == 2  # type: ignore
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_1.json'))
    assert not find_minimal_splitting_set(fbas2)
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_2.json'))
    assert find_minimal_splitting_set(fbas2)[0] == ['PK2']  # type: ignore


def test_min_splitting_set():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        config.card_encoding = 'totalizer'
        find_minimal_splitting_set(fbas_graph)


def test_min_blocking_set_3():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    config.card_encoding = 'totalizer'
    config.max_sat_algo = 'RC2'
    b = find_minimal_blocking_set(fbas1)
    assert len(b) == 2  # type: ignore


def test_min_blocking_set_4():
    data = get_test_data_list()
    for f, d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        config.card_encoding = 'totalizer'
        find_minimal_blocking_set(fbas_graph)


def test_min_quorum():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    assert len(find_min_quorum(fbas1)) == 3


def test_min_quorum_2():
    data = get_test_data_list()
    for f, d in data.items():
        if "validators" in f:
            logging.info("skipping %s", f)
            continue
        else:
            logging.info("loading graph of %s", f)
            fbas_graph = FBASGraph.from_json(d)
            find_min_quorum(fbas_graph)


def test_contains_quorum():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    assert contains_quorum({'PK1', 'PK2', 'PK3', 'PK4'}, fbas1)
    assert contains_quorum({'PK1', 'PK3', 'PK4'}, fbas1)
    assert not contains_quorum({'PK1', 'PK2'}, fbas1)
    assert not contains_quorum({'PK1'}, fbas1)
    assert not contains_quorum(set(), fbas1)
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_1.json'))
    assert contains_quorum({'PK1', 'PK2'}, fbas2)
    assert not contains_quorum({'PK1'}, fbas2)
    fbas2 = FBASGraph.from_json(
        get_validators_from_test_fbas('circular_2.json'))
    assert not contains_quorum({'PK1', 'PK2'}, fbas2)
    assert contains_quorum({'PK1', 'PK3'}, fbas2)


def test_top_tier():
    if HAS_QBF:
        qset1 = {
            'threshold': 3,
            'validators': [
                'PK1',
                'PK2',
                'PK3',
                'PK4'],
            'innerQuorumSets': []}
        fbas1 = FBASGraph()
        for v in ['PK1', 'PK2', 'PK3', 'PK4', 'PK5']:
            fbas1.update_validator(v, qset1)
        assert top_tier(fbas1) == {'PK1', 'PK2', 'PK3', 'PK4'}


def test_top_tier_2():
    if HAS_QBF:
        data = get_test_data_list()
        for f, d in data.items():
            if "validators" in f:
                logging.info("skipping %s", f)
                continue
            else:
                logging.info("loading graph of %s", f)
                fbas_graph = FBASGraph.from_json(d)
                top_tier(fbas_graph)


def test_is_overlay_resilient():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    g1 = nx.complete_graph(iter(['PK1', 'PK2', 'PK3', 'PK4']))
    assert is_overlay_resilient(fbas1, g1)
    g2 = nx.Graph()
    assert not is_overlay_resilient(fbas1, g2)
    g3 = nx.Graph()
    g3.add_edges_from([('PK1', 'PK2'), ('PK3', 'PK4')])
    assert not is_overlay_resilient(fbas1, g3)
    g4 = nx.Graph()
    g4.add_edges_from([('PK1', 'PK2'), ('PK2', 'PK3'), ('PK3', 'PK4')])
    assert not is_overlay_resilient(fbas1, g4)
    g5 = nx.Graph()
    g5.add_edges_from([('PK1', 'PK2'), ('PK2', 'PK3'),
                      ('PK3', 'PK4'), ('PK4', 'PK1')])
    assert is_overlay_resilient(fbas1, g5)


def test_num_not_blocked():
    qset1 = {
        'threshold': 3,
        'validators': [
            'PK1',
            'PK2',
            'PK3',
            'PK4'],
        'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1', 'PK2', 'PK3', 'PK4']:
        fbas1.update_validator(v, qset1)
    g1 = nx.complete_graph(iter(['PK1', 'PK2', 'PK3', 'PK4']))
    assert num_not_blocked(fbas1, g1) == 0
    g2 = nx.Graph()
    assert num_not_blocked(fbas1, g2) == 4
    g3 = nx.Graph()
    g3.add_edges_from([('PK1', 'PK2'), ('PK3', 'PK4')])
    assert num_not_blocked(fbas1, g3) == 4
    g4 = nx.Graph()
    g4.add_edges_from([('PK1', 'PK2'), ('PK2', 'PK3'), ('PK3', 'PK4')])
    assert num_not_blocked(fbas1, g4) == 2
    g5 = nx.Graph()
    g5.add_edges_from([('PK1', 'PK2'), ('PK2', 'PK3'),
                      ('PK3', 'PK4'), ('PK4', 'PK1')])
    assert num_not_blocked(fbas1, g5) == 0
