"""
Serialization and deserialization utilities for FBAS graphs.

This module provides a unified interface for handling conversion between 
FBASGraph objects and various JSON formats (stellarbeat.io and python-fbas formats).
"""

import json
import logging

from python_fbas.fbas_graph import FBASGraph
import python_fbas.stellarbeat_serializer as stellarbeat
import python_fbas.python_fbas_serializer as python_fbas_serializer


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


def serialize(fbas: FBASGraph, format: str = 'python-fbas') -> str:
    """
    Serialize an FBASGraph to JSON string.

    Args:
        fbas: The FBASGraph to serialize
        format: Target format ('python-fbas' or 'stellarbeat')
    """
    if format == 'python-fbas':
        return python_fbas_serializer.serialize(fbas)
    elif format == 'stellarbeat':
        serializer = stellarbeat.StellarBeatSerializer(fbas)
        return serializer.serialize(fbas)
    else:
        raise ValueError(f"Unknown format: {format}")

def deserialize(
        json_str: str,
        ignore_non_validating: bool = False) -> FBASGraph:
    """
    Deserialize JSON string to FBASGraph with automatic format detection.

    Args:
        json_str: JSON string to deserialize
        ignore_non_validating: If True, ignore non-validating validators (stellarbeat only)
    """
    # Parse JSON to detect format
    data = json.loads(json_str)
    format_type = detect_format(data)

    logging.info("Detected JSON format: %s", format_type)

    if format_type == 'stellarbeat':
        return stellarbeat.deserialize(json_str, ignore_non_validating)
    elif format_type == 'python-fbas':
        return python_fbas_serializer.deserialize(json_str)
    else:
        raise ValueError(
            f"Unknown or unsupported JSON format. Data type: {type(data)}")
