"""
Python-FBAS format serialization utilities for FBAS graphs.

This module provides the PythonFBASSerializer class to handle conversion 
between FBASGraph objects and the compact python-fbas JSON format.
"""

import json
import logging

from python_fbas.fbas_graph import FBASGraph


def serialize(fbas: FBASGraph) -> str:
    """
    Serialize the FBASGraph to python-fbas JSON format.

    Returns a JSON string representing the graph in a compact format.
    """
    # Collect validator data
    validators_data = []
    for v in fbas.validators:
        attrs = fbas.vertice_attrs(v).copy()

        # Remove quorum set related fields to avoid duplication
        # since we represent quorum sets separately in the qsets section
        attrs.pop('quorumSet', None)
        attrs.pop('quorumSetHashKey', None)

        qset_id = None
        if fbas.graph.out_degree(v) == 1:
            qset_id = fbas.qset_vertex_of(v)

        validators_data.append({
            "id": v,
            "qset": qset_id,
            "attrs": attrs
        })

    # Collect qset data
    qsets_data = {}
    qset_nodes = [
        q for q in fbas.graph.nodes() if q not in fbas.validators]
    for qset_id in qset_nodes:
        if qset_id in fbas.graph:
            threshold = fbas.threshold(qset_id)
            members = list(fbas.graph.successors(qset_id))
            qsets_data[qset_id] = {
                "threshold": threshold,
                "members": members
            }

    result = {
        "validators": validators_data,
        "qsets": qsets_data
    }

    return json.dumps(result, indent=2)


def deserialize(json_str: str) -> FBASGraph:
    """
    Create a FBASGraph from the python-fbas JSON format.

    Args:
        json_str: JSON string in python-fbas format
        ignore_non_validating: Not used for python-fbas format

    TODO: merge qsets with same threshold and successors
    """
    data = json.loads(json_str)

    if not isinstance(data, dict):
        raise ValueError("JSON data must be a dictionary")

    if "validators" not in data or "qsets" not in data:
        raise ValueError(
            "JSON data must contain 'validators' and 'qsets' keys")

    fbas = FBASGraph()

    # Check for duplicate IDs across validators and qsets
    all_ids = set()
    duplicates = set()

    # Check validator IDs
    for v_data in data["validators"]:
        if isinstance(v_data, dict) and "id" in v_data:
            validator_id = v_data["id"]
            if validator_id in all_ids:
                duplicates.add(validator_id)
            all_ids.add(validator_id)

    # Check qset IDs
    for qset_id in data["qsets"]:
        if qset_id in all_ids:
            duplicates.add(qset_id)
        all_ids.add(qset_id)

    if duplicates:
        raise ValueError(f"Duplicate IDs found: {sorted(duplicates)}")

    # First pass: add all validators
    for v_data in data["validators"]:
        if not isinstance(v_data, dict):
            logging.warning("Skipping invalid validator data: %s", v_data)
            continue

        if "id" not in v_data:
            logging.warning("Skipping validator without id: %s", v_data)
            continue

        validator_id = v_data["id"]
        attrs = v_data.get("attrs", {})

        fbas.add_validator(validator_id)
        if attrs:
            fbas.graph.nodes[validator_id].update(attrs)

    # Second pass: add qsets
    for qset_id, qset_data in data["qsets"].items():
        if not isinstance(qset_data, dict):
            logging.warning(
                "Skipping invalid qset data for %s: %s",
                qset_id,
                qset_data)
            continue

        if "threshold" not in qset_data or "members" not in qset_data:
            logging.warning(
                "Skipping qset %s missing threshold or members", qset_id)
            continue

        threshold = qset_data["threshold"]
        members = qset_data["members"]

        if not isinstance(threshold, int) or threshold < 0:
            logging.warning(
                "Skipping qset %s with invalid threshold: %s",
                qset_id,
                threshold)
            continue

        if not isinstance(members, list):
            logging.warning(
                "Skipping qset %s with invalid members: %s",
                qset_id,
                members)
            continue

        if threshold > len(members):
            logging.warning(
                "Skipping qset %s with threshold > members count", qset_id)
            continue

        # Add qset node
        fbas.graph.add_node(qset_id, threshold=threshold)

    # Third pass: add the edges
    for v_data in data["validators"]:
        if "id" not in v_data:
            continue

        validator_id = v_data["id"]
        qset_id = v_data.get("qset")

        if qset_id and qset_id in fbas.graph:
            fbas.graph.add_edge(validator_id, qset_id)

    for qset_id, qset_data in data["qsets"].items():
        for member in qset_data.get("members", []):
            if member in fbas.graph:
                # Add edge from qset to its members
                fbas.graph.add_edge(qset_id, member)
            else:
                logging.warning(
                    "Member %s of qset %s not found in graph", member, qset_id)

    fbas.check_integrity()

    return fbas
