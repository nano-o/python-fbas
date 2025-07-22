"""
Serialization and deserialization utilities for FBAS graphs.

This module provides classes to handle conversion between FBASGraph objects
and various JSON formats (stellarbeat.io and python-fbas formats).
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional, Set
from abc import ABC, abstractmethod

from python_fbas.fbas_graph import FBASGraph, QSet


class FBASSerializerBase(ABC):
    """Base class for FBAS serializers."""
    
    @abstractmethod
    def serialize(self, fbas: FBASGraph) -> str:
        """Serialize an FBASGraph to JSON string."""
        pass
    
    @abstractmethod
    def deserialize(self, json_str: str, ignore_non_validating: bool = False) -> FBASGraph:
        """Deserialize JSON string to FBASGraph."""
        pass


class StellarBeatSerializer(FBASSerializerBase):
    """Handles serialization/deserialization for stellarbeat.io format."""
    
    def serialize(self, fbas: FBASGraph) -> str:
        """
        Serialize the FBASGraph to stellarbeat.io JSON format.

        Returns a JSON string representing the graph as a list of validators
        in the stellarbeat format.
        """
        validators_list = []

        for v in sorted(fbas.validators):  # Sort for consistent output
            # Get all validator attributes
            attrs = fbas.vertice_attrs(v).copy()

            # The publicKey is the validator ID
            attrs['publicKey'] = v

            # Get the quorum set if it exists
            qset = fbas.qset_of(v)
            if qset is not None:
                attrs['quorumSet'] = self._qset_to_stellarbeat_format(qset)

            validators_list.append(attrs)

        return json.dumps(validators_list, indent=2)
    
    def _qset_to_stellarbeat_format(self, qset: QSet) -> Dict[str, Any]:
        """Convert a QSet to stellarbeat.io format."""
        result = {
            "threshold": qset.threshold,
            "validators": list(qset.validators),
            "innerQuorumSets": []
        }

        # Recursively convert inner quorum sets
        for inner_qset in qset.inner_quorum_sets:
            inner_qset_list = result["innerQuorumSets"]
            inner_qset_list.append(
                self._qset_to_stellarbeat_format(inner_qset))

        return result
    
    def deserialize(self, json_str: str, ignore_non_validating: bool = False) -> FBASGraph:
        """
        Create a FBASGraph from stellarbeat.io format JSON.
        
        Args:
            json_str: JSON string in stellarbeat format
            ignore_non_validating: If True, ignore non-validating validators
        """
        data = json.loads(json_str)
        
        if not isinstance(data, list):
            raise ValueError("Stellarbeat format expects a list of validators")
        
        # first do some validation
        validators = []
        keys = set()
        for v in data:
            if not isinstance(v, dict):
                logging.debug("Ignoring non-dict entry: %s", v)
                continue
            if 'publicKey' not in v:
                logging.debug(
                    "Entry is missing publicKey, skipping: %s", v)
                continue
            if (ignore_non_validating and (
                    ('isValidator' not in v or not v['isValidator'])
                    or ('isValidating' not in v or not v['isValidating']))):
                logging.debug(
                    "Ignoring non-validating validator: %s (name: %s)",
                    v['publicKey'],
                    v.get('name'))
                continue
            if 'quorumSet' not in v or v['quorumSet'] is None:
                logging.debug(
                    "Skipping validator missing quorumSet: %s",
                    v['publicKey'])
                continue
            if v['publicKey'] in keys:
                logging.warning(
                    "Ignoring duplicate validator: %s", v['publicKey'])
                continue
            keys.add(v['publicKey'])
            validators.append(v)
            
        # now create the graph:
        fbas = FBASGraph()
        qsets_dict: dict[str, QSet] = {}  # Local qsets dictionary for stellarbeat loading
        for v in validators:
            self._update_validator_from_json(fbas, v['publicKey'], qsets_dict, v['quorumSet'], v)

        fbas.check_integrity()
        return fbas
    
    def _update_validator_from_json(self, fbas: FBASGraph, v: Any, qsets_dict: dict[str, QSet], 
                                   qset: Optional[Dict[str, Any]] = None,
                                   attrs: Optional[Dict[str, Any]] = None) -> None:
        """
        Private method for loading validators from JSON format.
        """
        if attrs:
            # check that 'threshold' is not in attrs, as it's a reserved attribute
            if 'threshold' in attrs:
                raise ValueError(
                    "'threshold' is reserved and cannot be passed as an attribute")
            fbas.graph.add_node(v, **attrs)
        else:
            fbas.graph.add_node(v)
        fbas.validators.add(v)
        if qset:
            try:
                fqs = self._add_qset_from_json(fbas, qset, qsets_dict)
            except ValueError as e:
                logging.info(
                    "Failed to add qset for validator %s: %s. Qset data: %s", v, e, qset)
                return
            out_edges = list(fbas.graph.out_edges(v))
            fbas.graph.remove_edges_from(out_edges)
            fbas.graph.add_edge(v, fqs)
    
    def _add_qset_from_json(self, fbas: FBASGraph, qset: Dict[str, Any], 
                           qsets_dict: dict[str, QSet]) -> str:
        """
        Private method for loading qsets from JSON format.
        """
        match qset:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
                fqs = self._qset_from_json(qset)
                if fqs in qsets_dict.values():
                    return next(k for k, v in qsets_dict.items() if v == fqs)
                iqs_vertices = [self._add_qset_from_json(fbas, iq, qsets_dict) for iq in iqs]
                for v in vs:
                    fbas.add_validator(v)
                n = "_q" + uuid.uuid4().hex
                qsets_dict[n] = fqs
                fbas.graph.add_node(n, threshold=int(t))
                for member in set(vs) | set(iqs_vertices):
                    fbas.graph.add_edge(n, member)
                return n
            case _:
                logging.info(
                    "add_qset failed: qset does not match expected format. Expected keys: threshold, validators, innerQuorumSets. Actual keys: %s. Qset: %s",
                    list(
                        qset.keys()) if isinstance(
                        qset,
                        dict) else "not a dict",
                    qset)
                raise ValueError(
                    f"Invalid qset format (expected dict with threshold, validators, innerQuorumSets): {qset}")
    
    def _qset_from_json(self, qset: Dict[str, Any]) -> QSet:
        """
        Expects a JSON-serializable quorum-set (in stellarbeat.io format) and
        returns a QSet instance.
        """
        match qset:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqs}:
                threshold = int(t)
                validators = frozenset(vs)
                inner_qsets = frozenset(self._qset_from_json(iq) for iq in iqs)
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


class PythonFBASSerializer(FBASSerializerBase):
    """Handles serialization/deserialization for python-fbas format."""
    
    def serialize(self, fbas: FBASGraph) -> str:
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
        qset_nodes = [q for q in fbas.graph.nodes() if q not in fbas.validators]
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
    
    def deserialize(self, json_str: str, ignore_non_validating: bool = False) -> FBASGraph:
        """
        Create a FBASGraph from the python-fbas JSON format.
        
        Args:
            json_str: JSON string in python-fbas format
            ignore_non_validating: Not used for python-fbas format
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


class FBASSerializer:
    """
    Main serializer class with format auto-detection.
    
    This is the primary interface for serializing and deserializing FBAS graphs.
    """
    
    def __init__(self):
        self.stellarbeat = StellarBeatSerializer()
        self.python_fbas = PythonFBASSerializer()
    
    @staticmethod
    def detect_format(data) -> str:
        """
        Detect the format of JSON data for FBAS.

        Returns:
            'stellarbeat': Traditional stellarbeat.io format (list of validators)
            'python-fbas': New efficient python-fbas format (dict with validators/qsets)
            'unknown': Unable to determine format
        """
        if isinstance(data, list):
            # Stellarbeat format is a list of validator objects
            # If it's an empty list, assume stellarbeat format
            if len(data) == 0:
                return 'stellarbeat'

            # Check if all items are dictionaries (which would be validators)
            if all(isinstance(item, dict) for item in data):
                # For stellarbeat format, we expect most validators to have 'publicKey'
                # Check if at least some validators have this key
                validators_with_publickey = sum(
                    1 for item in data if 'publicKey' in item)
                if validators_with_publickey > 0:
                    return 'stellarbeat'

            return 'unknown'

        elif isinstance(data, dict):
            # Python-fbas format is a dict with 'validators' and 'qsets' keys
            if 'validators' in data and 'qsets' in data:
                # Additional validation: validators should be a list, qsets
                # should be a dict
                if isinstance(
                        data['validators'],
                        list) and isinstance(
                        data['qsets'],
                        dict):
                    return 'python-fbas'
            return 'unknown'

        return 'unknown'
    
    def serialize(self, fbas: FBASGraph, format: str = 'python-fbas') -> str:
        """
        Serialize an FBASGraph to JSON string.
        
        Args:
            fbas: The FBASGraph to serialize
            format: Target format ('python-fbas' or 'stellarbeat')
        """
        if format == 'python-fbas':
            return self.python_fbas.serialize(fbas)
        elif format == 'stellarbeat':
            return self.stellarbeat.serialize(fbas)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def deserialize(self, json_str: str, ignore_non_validating: bool = False) -> FBASGraph:
        """
        Deserialize JSON string to FBASGraph with automatic format detection.
        
        Args:
            json_str: JSON string to deserialize
            ignore_non_validating: If True, ignore non-validating validators (stellarbeat only)
        """
        # Parse JSON to detect format
        data = json.loads(json_str)
        format_type = self.detect_format(data)
        
        logging.info("Detected JSON format: %s", format_type)
        
        if format_type == 'stellarbeat':
            return self.stellarbeat.deserialize(json_str, ignore_non_validating)
        elif format_type == 'python-fbas':
            return self.python_fbas.deserialize(json_str, ignore_non_validating)
        else:
            raise ValueError(
                f"Unknown or unsupported JSON format. Data type: {type(data)}")