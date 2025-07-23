"""
Stellarbeat.io format serialization utilities for FBAS graphs.

This module provides the StellarBeatSerializer class to handle conversion
between FBASGraph objects and the stellarbeat.io JSON format.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional
from functools import lru_cache

from python_fbas.fbas_graph import FBASGraph
from python_fbas.config import get as get_config


@dataclass(frozen=True)
class QSet:
    """
    Represents a stellar-core quorum set in a unique, hashable way. Note that a
    quorum set is _not_ a set of quorums.  Instead, a quorum set represents
    agreement requirements. For quorums, see `is_quorum` in `FBASGraph`.
    """
    threshold: int
    validators: frozenset[str]
    inner_quorum_sets: frozenset['QSet']

    @staticmethod
    def from_json(qset: Dict[str, Any]) -> 'QSet':
        """
        Expects a JSON-serializable quorum-set (in stellarbeat.io format) and
        returns a QSet instance.
        """
        match qset:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
                threshold = int(t)
                validators = frozenset(vs)
                inner_qsets = frozenset(QSet.from_json(iq) for iq in iqs)
                card = len(validators) + len(inner_qsets)
                if not (0 <= threshold <= card):
                    logging.warning(
                        "QSet validation failed: threshold=%d not in range [0, %d] for qset with %d validators and %d inner quorum sets. Qset: %s",
                        threshold,
                        card,
                        len(validators),
                        len(inner_qsets),
                        qset)
                    raise ValueError(
                        f"Invalid qset threshold {threshold} (must be 0 <= threshold <= {card}): {qset}")
                return QSet(threshold, validators, inner_qsets)
            case _:
                logging.warning(
                    "QSet.from_json failed: qset does not match expected format. Expected keys: threshold, validators, innerQuorumSets. Actual keys: %s. Qset: %s",
                    list(
                        qset.keys()) if isinstance(
                        qset,
                        dict) else "not a dict",
                    qset)
                raise ValueError(
                    f"Invalid qset format (expected dict with threshold, validators, innerQuorumSets): {qset}")

    def to_json(self) -> Dict[str, Any]:
        """Convert a QSet to stellarbeat.io format."""
        result = {
            "threshold": self.threshold,
            "validators": list(self.validators),
            "innerQuorumSets": []
        }

        # Recursively convert inner quorum sets
        for inner_qset in self.inner_quorum_sets:
            result["innerQuorumSets"].append(inner_qset.to_json())

        return result


def deserialize(
        json_str: str,
        ignore_non_validating: bool = False) -> FBASGraph:
    """
    Create a FBASGraph from stellarbeat.io format JSON.

    Args:
        json_str: JSON string in stellarbeat format
        ignore_non_validating: If True, ignore non-validating validators

    TODO: handle ignore_non_validating
    """
    data = json.loads(json_str)

    if not isinstance(data, list):
        raise ValueError("Stellarbeat format expects a list of validators")

    # first do some validation
    validators = []
    keys = set()
    config = get_config()
    for v in data:
        if not isinstance(v, dict) or 'publicKey' not in v:
            if config.deserialization_mode == "indulgent":
                logging.warning("Ignoring invalid validator format: %s", v)
                continue
            else:
                raise ValueError(f"Invalid validator format (expected dict with 'publicKey'): {v}")

        pk = v['publicKey']
        if pk in keys:
            if config.deserialization_mode == "indulgent":
                logging.warning("Ignoring duplicate validator: %s", pk)
                continue
            else:
                raise ValueError(f"Duplicate validator publicKey: {pk}")

        keys.add(pk)
        validators.append(v)

    # now create the graph:
    fbas = FBASGraph()
    qsets: dict[QSet, str] = {}

    def _process_qset(qset: QSet) -> str:
        if qset in qsets:
            return qsets[qset]
        else:
            qset_id = "_q" + uuid.uuid4().hex
            qsets[qset] = qset_id
            # add the qset to the graph
            fbas.graph.add_node(qset_id, threshold=int(qset.threshold))
            for pk in qset.validators:
                keys.add(pk)
                fbas.graph.add_edge(qset_id, pk)
            for iqs in qset.inner_quorum_sets:
                iqs_id = _process_qset(iqs)
                fbas.graph.add_edge(qset_id, iqs_id)
            return qset_id

    for v in validators:
        # remove 'quorumSet' from the validator attributes if it exists:
        attrs = v.copy()
        attrs.pop('quorumSet', None)
        pk = v['publicKey']
        fbas.graph.add_node(pk, **attrs)
        qset_json = v.get('quorumSet', None)
        if qset_json is not None:
            try:
                qset = QSet.from_json(qset_json)
                qset_id = _process_qset(qset)
                fbas.graph.add_edge(pk, qset_id)
            except ValueError as e:
                if get_config().deserialization_mode == "indulgent":
                    logging.warning(f"Skipping invalid quorum set for validator {pk}: {e}")
                else:
                    raise

    fbas.validators = keys

    fbas.check_integrity()
    return fbas


@lru_cache(maxsize=128)
def compute_qset(fbas: FBASGraph, qset_vertex: str) -> QSet:
    """
    Recursively computes the QSet associated with the given qset vertex.
    """
    assert qset_vertex not in fbas.validators
    threshold = fbas.threshold(qset_vertex)
    # validators are the children of the qset vertex that are validators:
    validators = frozenset(v for v in fbas.graph.successors(
        qset_vertex) if v in fbas.validators)
    # inner_qsets are the children of the qset vertex that are qset
    # vertices:
    inner_qsets = frozenset(compute_qset(fbas, q) for q in fbas.graph.successors(
        qset_vertex) if q not in fbas.validators)
    return QSet(threshold, validators, inner_qsets)


def qset_of(fbas: FBASGraph, n: str) -> Optional[QSet]:
    """
    Computes the QSet associated with the given vertex n based on the graph (does not use the qsets dict).
    n must be a validator vertex.
    """
    assert n in fbas.validators
    # if n has no successors, then we don't know its qset:
    if fbas.graph.out_degree(n) == 0:
        return None
    return compute_qset(fbas, fbas.qset_vertex_of(n))


class StellarBeatSerializer:
    """Handles serialization for stellarbeat.io format."""
    fbas: FBASGraph

    def __init__(self, fbas: FBASGraph):
        self.fbas = fbas

    def serialize(self) -> str:
        """
        Serialize the FBASGraph to stellarbeat.io JSON format.

        Returns a JSON string representing the graph as a list of validators
        in the stellarbeat format.
        """
        validators_list = []

        for v in sorted(self.fbas.validators):  # Sort for consistent output
            # Get all validator attributes
            attrs = self.fbas.vertice_attrs(v).copy()

            # If publicKey is not present, then it's the validator ID:
            if 'publicKey' not in attrs:
                attrs['publicKey'] = v

            # Get the quorum set if it exists
            qset = qset_of(self.fbas, v)
            if qset is not None:
                attrs['quorumSet'] = qset.to_json()

            validators_list.append(attrs)

        return json.dumps(validators_list, indent=2)
