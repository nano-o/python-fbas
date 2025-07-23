"""
Python-FBAS format serialization utilities for FBAS graphs.

This module provides the PythonFBASSerializer class to handle conversion
between FBASGraph objects and the compact python-fbas JSON format.

TODO: this is quick and dirty, needs to be cleaned up and tested properly.
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
    for v in fbas.get_validators():
        attrs = fbas.vertice_attrs(v).copy()

        # Remove quorum set related fields to avoid duplication
        # since we represent quorum sets separately in the qsets section
        attrs.pop('quorumSet', None)
        attrs.pop('quorumSetHashKey', None)

        qset_id = None
        if fbas.get_out_degree(v) == 1:
            qset_id = fbas.qset_vertex_of(v)

        validators_data.append({
            "id": v,
            "qset": qset_id,
            "attrs": attrs
        })

    # Collect qset data
    qsets_data = {}
    qset_nodes = [
        q for q in fbas.vertices() if not fbas.is_validator(q)]
    for qset_id in qset_nodes:
        if fbas.has_vertex(qset_id):
            threshold = fbas.threshold(qset_id)
            members = fbas.get_successors(qset_id)
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

    TODO: merge qsets with same threshold and successors
    """
    data = json.loads(json_str)

    # start with consistency checks
    if not isinstance(data, dict):
        raise ValueError("JSON data must be a dictionary")

    if "validators" not in data or "qsets" not in data:
        raise ValueError(
            "JSON data must contain 'validators' and 'qsets' keys")

    fbas = FBASGraph()

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

        fbas.add_validator(validator_id, qset=None, **attrs)

    # TODO check for cycles...

    # Third pass: add qsets
    def _add_qset(qid, qset):
        threshold = qset["threshold"]
        members = qset["members"]
        # assumin all validators have been added already...
        for member in members:
            if (not fbas.is_validator(member)) and member not in fbas.vertices():
                _add_qset(member, data['qsets'][member])
        fbas.add_qset(threshold, members, qset_id=qid)

    for qset_id, qset in data["qsets"].items():
        _add_qset(qset_id, qset)

    # Fourth pass: connect validators to their qsets
    for v_data in data["validators"]:
        if "id" not in v_data:
            continue

        validator_id = v_data["id"]
        qset_id = v_data.get("qset")

        if qset_id and fbas.has_vertex(qset_id):
            fbas.update_validator(validator_id, qset=qset_id)

    fbas.check_integrity()

    return fbas
