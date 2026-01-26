"""Tests for org_graph module."""

import pytest

from python_fbas.fbas_graph import FBASGraph
from python_fbas.org_graph import fbas_to_org_graph
from python_fbas.serialization import deserialize


class TestFbasToOrgGraph:
    """Tests for fbas_to_org_graph function."""

    def test_simple_two_org_structure(self):
        """Test a simple org-structured FBAS with 2 orgs, each with 2 validators."""
        fbas = FBASGraph()

        # Org 1: validators v1, v2
        fbas.add_validator('v1')
        fbas.add_validator('v2')
        # Org 2: validators v3, v4
        fbas.add_validator('v3')
        fbas.add_validator('v4')

        # Create org qsets (leaf qsets containing only validators)
        org1_qset = fbas.add_qset(2, ['v1', 'v2'])
        org2_qset = fbas.add_qset(2, ['v3', 'v4'])

        # Each validator's qset points to both orgs
        validator_qset = fbas.add_qset(2, [org1_qset, org2_qset])
        fbas.update_validator('v1', qset=validator_qset)
        fbas.update_validator('v2', qset=validator_qset)
        fbas.update_validator('v3', qset=validator_qset)
        fbas.update_validator('v4', qset=validator_qset)

        result = fbas_to_org_graph(fbas)

        assert result is not None
        assert set(result.nodes()) == {org1_qset, org2_qset}
        # Both orgs point to each other since their validators' qset contains both org qsets
        assert set(result.edges()) == {(org1_qset, org2_qset), (org2_qset, org1_qset)}

    def test_single_org(self):
        """Test an FBAS with a single organization."""
        fbas = FBASGraph()

        fbas.add_validator('v1')
        fbas.add_validator('v2')

        org_qset = fbas.add_qset(2, ['v1', 'v2'])

        # Validators' qset is just the org qset itself
        fbas.update_validator('v1', qset=org_qset)
        fbas.update_validator('v2', qset=org_qset)

        result = fbas_to_org_graph(fbas)

        assert result is not None
        assert set(result.nodes()) == {org_qset}
        assert len(result.edges()) == 0

    def test_three_orgs_with_nested_structure(self):
        """Test 3 orgs where one org references others through nested qsets."""
        fbas = FBASGraph()

        # Create validators for 3 orgs
        for i in range(1, 7):
            fbas.add_validator(f'v{i}')

        # Org qsets
        org1_qset = fbas.add_qset(2, ['v1', 'v2'])
        org2_qset = fbas.add_qset(2, ['v3', 'v4'])
        org3_qset = fbas.add_qset(2, ['v5', 'v6'])

        # Org1 validators' qset references org2 and org3
        v1_qset = fbas.add_qset(2, [org1_qset, org2_qset, org3_qset])
        fbas.update_validator('v1', qset=v1_qset)
        fbas.update_validator('v2', qset=v1_qset)

        # Org2 validators' qset references org1 and org3
        v3_qset = fbas.add_qset(2, [org1_qset, org2_qset, org3_qset])
        fbas.update_validator('v3', qset=v3_qset)
        fbas.update_validator('v4', qset=v3_qset)

        # Org3 validators' qset references org1 and org2
        v5_qset = fbas.add_qset(2, [org1_qset, org2_qset, org3_qset])
        fbas.update_validator('v5', qset=v5_qset)
        fbas.update_validator('v6', qset=v5_qset)

        result = fbas_to_org_graph(fbas)

        assert result is not None
        assert set(result.nodes()) == {org1_qset, org2_qset, org3_qset}
        # Each org points to the other two
        assert result.out_degree(org1_qset) == 2
        assert result.out_degree(org2_qset) == 2
        assert result.out_degree(org3_qset) == 2

    def test_validator_missing_qset_returns_none(self):
        """Test that FBAS with validator missing qset returns None when check_same_qset=True."""
        fbas = FBASGraph()

        fbas.add_validator('v1')
        fbas.add_validator('v2')
        fbas.add_validator('v3')

        org_qset = fbas.add_qset(2, ['v1', 'v2', 'v3'])
        fbas.update_validator('v1', qset=org_qset)
        fbas.update_validator('v2', qset=org_qset)
        # v3 has no qset assigned

        result = fbas_to_org_graph(fbas, check_same_qset=True)

        assert result is None

    def test_validators_with_different_qsets_returns_none(self):
        """Test that validators in same org with different qsets returns None when check_same_qset=True."""
        fbas = FBASGraph()

        fbas.add_validator('v1')
        fbas.add_validator('v2')
        fbas.add_validator('v3')

        org_qset = fbas.add_qset(2, ['v1', 'v2'])

        # Create two different qsets for the validators by adding different additional members
        extra_qset = fbas.add_qset(1, ['v3'])
        q1 = fbas.add_qset(1, [org_qset])
        q2 = fbas.add_qset(2, [org_qset, extra_qset])  # Different structure

        fbas.update_validator('v1', qset=q1)
        fbas.update_validator('v2', qset=q2)
        fbas.update_validator('v3', qset=q1)

        result = fbas_to_org_graph(fbas, check_same_qset=True)

        assert result is None

    def test_validator_in_multiple_qsets_returns_none(self):
        """Test that validator appearing in multiple qsets returns None."""
        fbas = FBASGraph()

        fbas.add_validator('v1')
        fbas.add_validator('v2')
        fbas.add_validator('v3')

        # v1 and v2 in org1
        org1_qset = fbas.add_qset(2, ['v1', 'v2'])
        # v2 and v3 in org2 - v2 is in both!
        org2_qset = fbas.add_qset(2, ['v2', 'v3'])

        validator_qset = fbas.add_qset(2, [org1_qset, org2_qset])
        fbas.update_validator('v1', qset=validator_qset)
        fbas.update_validator('v2', qset=validator_qset)
        fbas.update_validator('v3', qset=validator_qset)

        result = fbas_to_org_graph(fbas)

        assert result is None

    def test_deeply_nested_qset_structure(self):
        """Test org-graph edges with deeply nested qset references."""
        fbas = FBASGraph()

        fbas.add_validator('v1')
        fbas.add_validator('v2')
        fbas.add_validator('v3')
        fbas.add_validator('v4')

        org1_qset = fbas.add_qset(2, ['v1', 'v2'])
        org2_qset = fbas.add_qset(2, ['v3', 'v4'])

        # Create a nested structure: intermediate -> org2
        intermediate_qset = fbas.add_qset(1, [org2_qset])
        # Org1's validators' qset contains intermediate (which contains org2)
        v1_qset = fbas.add_qset(2, [org1_qset, intermediate_qset])
        fbas.update_validator('v1', qset=v1_qset)
        fbas.update_validator('v2', qset=v1_qset)

        # Org2's validators' qset directly contains org1
        v3_qset = fbas.add_qset(2, [org1_qset, org2_qset])
        fbas.update_validator('v3', qset=v3_qset)
        fbas.update_validator('v4', qset=v3_qset)

        result = fbas_to_org_graph(fbas)

        assert result is not None
        assert set(result.nodes()) == {org1_qset, org2_qset}
        # Org1 -> Org2 (through nested intermediate)
        assert (org1_qset, org2_qset) in result.edges()
        # Org2 -> Org1 (direct)
        assert (org2_qset, org1_qset) in result.edges()

    def test_empty_fbas(self):
        """Test that empty FBAS returns empty org-graph."""
        fbas = FBASGraph()

        result = fbas_to_org_graph(fbas)

        assert result is not None
        assert len(result.nodes()) == 0
        assert len(result.edges()) == 0

    def test_org_graph_vertices_are_strings(self):
        """Test that org-graph vertices are qset ID strings."""
        fbas = FBASGraph()

        fbas.add_validator('v1')
        fbas.add_validator('v2')

        org_qset = fbas.add_qset(2, ['v1', 'v2'])
        fbas.update_validator('v1', qset=org_qset)
        fbas.update_validator('v2', qset=org_qset)

        result = fbas_to_org_graph(fbas)

        assert result is not None
        for node in result.nodes():
            assert isinstance(node, str)
            assert node == org_qset

    def test_top_tier_json_is_org_structured(self):
        """Test that the main top_tier.json file is org-structured."""
        with open('python_fbas/data/top_tier.json') as f:
            fbas = deserialize(f.read())

        result = fbas_to_org_graph(fbas)

        assert result is not None
        # top_tier.json has 21 validators in 7 orgs
        assert len(fbas.get_validators()) == 21
        assert len(result.nodes()) == 7
        # Fully connected: each org references all 6 others
        assert len(result.edges()) == 42

    def test_small_top_tier_json_is_org_structured(self):
        """Test that the small test top_tier.json file is org-structured."""
        with open('tests/test_data/small/top_tier.json') as f:
            fbas = deserialize(f.read())

        result = fbas_to_org_graph(fbas)

        assert result is not None
        # small top_tier.json has 23 validators in 7 orgs
        assert len(fbas.get_validators()) == 23
        assert len(result.nodes()) == 7
        # Fully connected: each org references all 6 others
        assert len(result.edges()) == 42
