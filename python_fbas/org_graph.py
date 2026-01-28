"""
We expect an FBAS to be structured in terms of organizations (orgs for short).
An organization is a set of validators such that:
- They all have the same qset Q
- They are direct members of a unique qset
If we can partition the validators into organizations, we say that the FBAS is org-structured.
"""

from __future__ import annotations

import networkx as nx

from python_fbas.fbas_graph import FBASGraph

def fbas_to_org_graph(fbas: FBASGraph) -> nx.DiGraph | None:
    """
    If the FBAS is not org-structured, return None.
    Otherwise, return the org-graph representation of the FBAS:
    - Each vertex in the org-graph is the ID of an organization qset from the FBAS.
    - There is a directed edge from org A to org B if and only if the qset of the validators of org A contains (as inner qset, but maybe deeply nested) or is the qset of org B.

    Args:
        fbas: The FBAS graph to analyze.
    """
    graph = fbas.graph_view()
    validators = fbas.get_validators()
    qset_vertices = set(fbas.get_qset_vertices())

    # Step 1: Find all leaf qsets (qsets with no inner qsets - all successors are validators)
    # These are the only possible org qsets
    leaf_qsets: set[str] = set()
    for q in qset_vertices:
        successors = set(graph.successors(q))
        if successors and all(s in validators for s in successors):
            leaf_qsets.add(q)

    # Step 2: Build a mapping from each validator to all qsets that directly contain it
    validator_to_containing_qsets: dict[str, set[str]] = {v: set() for v in validators}
    for q in qset_vertices:
        for succ in graph.successors(q):
            if succ in validators:
                validator_to_containing_qsets[succ].add(q)

    # Step 3: Verify org-structure and identify org qsets
    # For each leaf qset, check:
    # - All validators in the qset have the same qset (pointing to the same qset vertex)
    # - Those validators do not appear in any other qset
    org_qsets: set[str] = set()
    validators_in_orgs: set[str] = set()

    for leaf_q in leaf_qsets:
        member_validators = [v for v in graph.successors(leaf_q) if v in validators]

        if not member_validators:
            continue

        # Check that all validators in this leaf qset have the same qset
        validator_qsets = [fbas.qset_vertex_of(v) for v in member_validators]

        # All validators must have a qset defined
        if any(q is None for q in validator_qsets):
            return None

        # All validators must have the same qset
        if len(set(validator_qsets)) != 1:
            return None

        # Check that these validators only appear in this leaf_q (and not in any other qset)
        for v in member_validators:
            containing_qsets = validator_to_containing_qsets[v]
            if containing_qsets != {leaf_q}:
                return None

        # Check for overlap with validators already assigned to an org
        for v in member_validators:
            if v in validators_in_orgs:
                return None
            validators_in_orgs.add(v)

        org_qsets.add(leaf_q)

    # Step 4: Check that all validators are accounted for
    if validators_in_orgs != validators:
        return None

    # Step 5: Build the org-graph
    org_graph = nx.DiGraph()

    # Helper function to check if a qset contains another qset (possibly deeply nested)
    def contains_qset(qset: str, target: str, visited: set[str] | None = None) -> bool:
        """Check if target qset is reachable from qset through qset vertices only."""
        if visited is None:
            visited = set()
        if qset in visited:
            return False
        visited.add(qset)

        for succ in graph.successors(qset):
            if succ == target:
                return True
            if succ in qset_vertices:
                if contains_qset(succ, target, visited):
                    return True
        return False

    # Step 6: Add edges between org qsets and collect threshold info
    # For each org A, find the qset of its validators and check if it contains org B
    org_thresholds: dict[str, int | None] = {}
    for org_a in org_qsets:
        # Get one validator from org_a to find the common qset of org_a's validators
        member_validators = [v for v in graph.successors(org_a) if v in validators]
        if not member_validators:
            continue

        validator_qset = fbas.qset_vertex_of(member_validators[0])
        if validator_qset is None:
            continue

        # Store the threshold of the validator's qset for this org
        org_thresholds[org_a] = graph.nodes[validator_qset].get('threshold')

        # Check if this validator's qset contains or equals any other org qset
        for org_b in org_qsets:
            if org_a == org_b:
                continue
            # Check if validator_qset contains org_b or is org_b
            if validator_qset == org_b or contains_qset(validator_qset, org_b):
                org_graph.add_edge(org_a, org_b)

    # Add vertices for each org qset with their threshold attributes
    for org_q in org_qsets:
        threshold = org_thresholds.get(org_q)
        if threshold is not None:
            org_graph.add_node(org_q, threshold=threshold)
        else:
            org_graph.add_node(org_q)

    return org_graph


def org_graph_to_fbas(org_graph: nx.DiGraph) -> FBASGraph:
    """
    Convert an org graph into an FBASGraph.

    For each org in the org graph:
    - Create 3 validators named {org}_0, {org}_1, {org}_2
    - Create an org qset containing those 3 validators with threshold 2
    - Give each validator a qset derived from the org graph edges:
      the qset contains the org qsets of the orgs that this org points to,
      with the threshold from the org graph node.

    Args:
        org_graph: A directed graph where each node represents an organization
                   and has a "threshold" attribute. Edges represent dependencies
                   between organizations.

    Returns:
        An FBASGraph representing the org-structured FBAS.
    """
    fbas = FBASGraph()

    # Step 1: Create 3 validators for each org
    for org in org_graph.nodes:
        for i in range(3):
            fbas.add_validator(f"{org}_{i}")

    # Step 2: Create org qsets (threshold 2, containing the 3 validators)
    org_to_org_qset: dict[str, str] = {}
    for org in org_graph.nodes:
        validators = [f"{org}_{i}" for i in range(3)]
        org_qset_id = fbas.add_qset(2, validators)
        org_to_org_qset[org] = org_qset_id

    # Step 3: Create validator qsets based on org graph edges and assign to validators
    for org in org_graph.nodes:
        out_orgs = list(org_graph.successors(org))
        if not out_orgs:
            raise ValueError(
                f"Org graph node {org} has no outgoing edges; cannot build qset")

        threshold = org_graph.nodes[org].get("threshold")
        if threshold is None:
            # Default to 2/3 (rounded up) of the number of children
            threshold = (2 * len(out_orgs) + 2) // 3

        if not isinstance(threshold, int):
            raise ValueError(
                f"Org graph node {org} has non-integer threshold {threshold}")

        if not 1 <= threshold <= len(out_orgs):
            raise ValueError(
                f"Org graph node {org} has threshold {threshold} outside "
                f"[1, {len(out_orgs)}]")

        # Build the validator qset from the org qsets of the target orgs
        target_org_qsets = [org_to_org_qset[target_org] for target_org in out_orgs]
        validator_qset_id = fbas.add_qset(threshold, target_org_qsets)

        # Assign this qset to all 3 validators of this org
        for i in range(3):
            fbas.update_validator(f"{org}_{i}", qset=validator_qset_id)

    return fbas


def org_graph_to_org_level_fbas(org_graph: nx.DiGraph) -> FBASGraph:
    """
    Convert an org graph into a simplified FBASGraph where each org is a single validator.

    Each org vertex becomes a validator whose quorum set is formed by its
    outgoing neighbors (as validators, not org qsets) and the vertex "threshold"
    attribute. This representation is useful for org-level analysis and for
    exporting a compact FBAS.

    Args:
        org_graph: A directed graph where each node represents an organization
                   and has a "threshold" attribute. Edges represent dependencies
                   between organizations.

    Returns:
        An FBASGraph where each org is a single validator.
    """
    fbas = FBASGraph()

    for org in org_graph.nodes:
        fbas.add_validator(org)

    for org in org_graph.nodes:
        out_orgs = list(org_graph.successors(org))
        if not out_orgs:
            raise ValueError(
                f"Org graph node {org} has no outgoing edges; cannot build qset")

        threshold = org_graph.nodes[org].get("threshold")
        if threshold is None:
            # Default to 2/3 (rounded up) of the number of children
            threshold = (2 * len(out_orgs) + 2) // 3

        if not isinstance(threshold, int):
            raise ValueError(
                f"Org graph node {org} has non-integer threshold {threshold}")

        if not 1 <= threshold <= len(out_orgs):
            raise ValueError(
                f"Org graph node {org} has threshold {threshold} outside "
                f"[1, {len(out_orgs)}]")

        qset_id = fbas.add_qset(threshold, out_orgs)
        fbas.update_validator(org, qset=qset_id)

    return fbas
