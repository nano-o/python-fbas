"""
Python-FBAS format serialization utilities for FBAS graphs.

Format overview:
- Top-level JSON object with two keys: "validators" (list) and "qsets" (dict).
- Each validator entry is {"id": <validator_id>, "qset": <qset_id|None>,
  "attrs": <dict>}. No duplicate IDs are allowed.
- "qsets" maps qset IDs to {"threshold": <int>, "members":
  [<validator_id|qset_id>, ...]}.
- Members can reference validators or other qsets by ID.
- QSets that have the same threshold and same set of members are considered the
  same. While such duplicates are not an error, only one will be kept when
  deserializing.
- Qset references must not form cycles.
- Validator ids and QSet ids must be disjoint.

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
    for v in fbas.get_validators():
        attrs = fbas.vertice_attrs(v).copy()

        # Remove quorum set related fields to avoid duplication
        # since we represent quorum sets separately in the qsets section
        attrs.pop('quorumSet', None)
        # Keep quorumSetHashKey as it's part of the validator's attributes

        qset_id = None
        if fbas.graph_view().out_degree(v) == 1:
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
        if qset_id in fbas.graph_view():
            threshold = fbas.threshold(qset_id)
            members = list(fbas.graph_view().successors(qset_id))
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

    If the input data contains duplicate qsets (same members and threshold but
    different keys), only the first one will be kept.
    """
    data = json.loads(json_str)

    # start with consistency checks
    if not isinstance(data, dict):
        raise ValueError("JSON data must be a dictionary")

    if "validators" not in data or "qsets" not in data:
        raise ValueError(
            "JSON data must contain 'validators' and 'qsets' keys")

    fbas = FBASGraph()

    # add all validators
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

    # add the qsets
    qset_id_map: dict[str, str] = {}

    def _add_qset(qid, qset, stack):
        # assuming all validators have been added already...
        if qid in qset_id_map:
            return qset_id_map[qid]
        if qid in stack:
            raise ValueError(f"Detected qset cycle involving: {qid}")
        if qid in fbas.vertices():
            if fbas.is_validator(qid):
                raise ValueError(f"QSet ID {qid} collides with a validator ID")
            qset_id_map[qid] = qid
            return qid

        stack.add(qid)
        threshold = qset["threshold"]
        members = qset["members"]
        remapped_members = []
        for member in members:
            if fbas.is_validator(member):
                remapped_members.append(member)
                continue
            if member not in fbas.vertices():
                _add_qset(member, data['qsets'][member], stack)
            remapped_members.append(qset_id_map.get(member, member))
        actual_id = fbas.add_qset(threshold, remapped_members, qset_id=qid)
        qset_id_map[qid] = actual_id
        stack.remove(qid)
        return actual_id

    for qset_id, qset in data["qsets"].items():
        _add_qset(qset_id, qset, set())

    # connect validators to their qsets
    for v_data in data["validators"]:
        if "id" not in v_data:
            continue

        validator_id = v_data["id"]
        qset_id = v_data.get("qset")

        if qset_id:
            qset_id = qset_id_map.get(qset_id, qset_id)
        if qset_id and qset_id in fbas.graph_view():
            fbas.update_validator(validator_id, qset=qset_id)

    fbas.check_integrity()

    return fbas
