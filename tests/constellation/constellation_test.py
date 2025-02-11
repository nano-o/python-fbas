import logging
from python_fbas.constellation.constellation import *

def test_regular_fbas_to_fbas_graph():
    regular_fbas_to_fbas_graph({'A':(1, ['A'])})
    regular_fbas_to_fbas_graph({'A':(2, ['A','B','C']), 'B':(2, ['A','B']), 'C':(1, ['A','B','C'])})

def test_compute_clusters():
    # regular fbas with 10 organizations, each with threshold 7:
    fbas = {f"O_{i}":7 for i in range(1,11)}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    assert clusters ==  [{'O_3', 'O_5', 'O_1', 'O_2', 'O_4'}, {'O_10', 'O_9', 'O_7', 'O_6', 'O_8'}]
    # regular fbas with 11 organizations, each with threshold 8:
    fbas = {f"O_{i}":8 for i in range(1,12)}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    assert clusters == [{'O_1', 'O_2', 'O_3', 'O_4'}, {'O_7', 'O_6', 'O_8', 'O_5'}, {'O_10', 'O_11', 'O_9'}]
    # non-uniform thresholds:
    t1 = 9
    t2 = 6
    t3 = 8
    fbas = {f"O_{i}":t1 for i in range(1,4)}\
        | {f"O_{i}":t2 for i in range(4,8)} | {f"O_{i}":t3 for i in range(8,12)}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    assert clusters == [{'O_7', 'O_6', 'O_5', 'O_4'}, {'O_9', 'O_11', 'O_8', 'O_10'}, {'O_2', 'O_1', 'O_3'}]

def test_compute_overlay():
    fbas = {'A':1, 'B':1}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    g = clusters_to_overlay(clusters)
    logging.info("overlay: %s", g.edges())
    assert len(g.edges()) == 12
    fbas = {'A':2, 'B':2, 'C':2}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    g = clusters_to_overlay(clusters)
    # logging.info("overlay: %s", g.edges())
    assert len(g.edges()) == 27
    fbas = {'A':2, 'B':2, 'C':2, 'D':2}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    g = clusters_to_overlay(clusters)
    # logging.info("overlay: %s", g.edges())
    assert len(g.edges()) == 42
    # regular fbas with 10 organizations, each with threshold 7:
    fbas = {f"O_{i}":7 for i in range(1,11)}
    g = clusters_to_overlay(compute_clusters(fbas))
    # logging.info("overlay: %s", g.edges())
    assert len(g.edges()) == 195
    fbas = {'A':1, 'B':1}
    clusters = compute_clusters(fbas)
    logging.info("clusters: %s", clusters)
    g = clusters_to_overlay(clusters, num_validators={'A':3, 'B':5})
    logging.info("overlay: %s", g.edges())
    assert len(g.edges()) == 25

def test_random_constellation_overlay():
    for _ in range(10):
        size = 11
        fbas = random_single_universe_regular_fbas(size, int(size/2), size)
        overlay = constellation_overlay(fbas)
        logging.info("number of edges: %s", len(overlay.edges()))

def test_overlay_from_file():
    file = 'tests/constellation/test_data/10_orgs_single_universe_regular_fbas.json'
    with open(file, 'r', encoding='utf-8') as f:
        fbas = json.load(f)
    constellation_overlay(fbas)

def test_greedy_overlay():
    fbas = {'A':1, 'B':1}
    g = greedy_overlay(fbas)
    fbas = {'A':2, 'B':2, 'C':2, 'D':2}
    g = greedy_overlay(fbas)
    fbas = {'A':2, 'B':2, 'C':1, 'D':2}
    g = greedy_overlay(fbas)
    fbas = {'A':2, 'B':4, 'C':1, 'D':2}
    g = greedy_overlay(fbas)
