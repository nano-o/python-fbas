"""
Stellarbeat.io format serialization utilities for FBAS graphs.

Stellarbeat format:
- A JSON array of validator objects.
- Each validator includes a "publicKey" and optional metadata.
- The quorum set is under "quorumSet" with:
  - "threshold": int
  - "validators": list of validator public keys
  - "innerQuorumSets": list of nested quorum sets in the same format
- Quorum sets are fully inlined per validator (no shared references), which
  can lead to substantial duplication when many validators share the same
  structure.

Deserialization:
- Each validator becomes a vertex with an edge to its qset.
- Each quorum set becomes a separate qset vertex with edges to its
  validators and inner qset vertices.
- Qset IDs are generated via FBASGraph.add_qset (or reused when an
  identical threshold+components qset already exists).
- Qsets are deduplicated: any quorum sets with the same threshold and
  component members (validators and inner qsets) map to a single qset
  vertex ID.

Serialization:
- Validators are emitted as objects with their attributes and "publicKey".
- The validator's associated qset vertex is traversed to rebuild the
  inline, nested "quorumSet" structure.
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
        inner_qsets = [inner.to_json() for inner in self.inner_quorum_sets]
        inner_qsets_sorted = sorted(
            inner_qsets,
            key=lambda qset: json.dumps(qset, sort_keys=True),
        )
        return {
            "threshold": self.threshold,
            "validators": sorted(self.validators),
            "innerQuorumSets": inner_qsets_sorted,
        }


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
    validator_home_domains: dict[str, str] = {}
    home_domain_counts: dict[str, int] = {}
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
        home_domain = v.get('homeDomain')
        if home_domain:
            validator_home_domains[pk] = home_domain
            home_domain_counts[home_domain] = home_domain_counts.get(
                home_domain, 0) + 1

    # now create the graph:
    fbas = FBASGraph()

    def _qset_id_for(qset: QSet) -> str | None:
        if qset.inner_quorum_sets or not qset.validators:
            return None
        # Use the homeDomain as the qset ID only when every member has one,
        # all members share the same homeDomain, and no other validator uses it.
        for pk in qset.validators:
            if pk not in validator_home_domains:
                return None
        qset_home_domains = {
            validator_home_domains[pk] for pk in qset.validators
        }
        if len(qset_home_domains) != 1:
            return None
        home_domain = next(iter(qset_home_domains))
        if home_domain_counts.get(home_domain) != len(qset.validators):
            return None
        if home_domain in fbas.graph_view():
            logging.warning(
                "homeDomain '%s' is already in use as a vertex ID; using generated qset ID instead",
                home_domain)
            return None
        return home_domain

    def _process_qset(qset: QSet) -> str:
        # we need to add the inner qsets bottom up:
        iqss = set()
        if qset.inner_quorum_sets:
            for inner_qset in qset.inner_quorum_sets:
                qis = _process_qset(inner_qset)
                iqss.add(qis)
        for pk in qset.validators:
            if not fbas.is_validator(pk):
                fbas.add_validator(pk)
        qset_id = _qset_id_for(qset)
        return fbas.add_qset(
            qset.threshold,
            qset.validators | iqss,
            qset_id=qset_id)

    for v in validators:
        attrs = v.copy()
        # remove the quorumSet attribute as it's verbose and will be encoded in
        # the graph:
        q = attrs.pop('quorumSet', None)
        pk = v['publicKey']
        fbas.update_validator(pk, qset=None, **attrs)
        if q:
            try:
                qset_id = _process_qset(QSet.from_json(q))
            except ValueError as e:
                if config.deserialization_mode == "indulgent":
                    logging.debug("Ignoring invalid quorum set: %s", e)
                    continue
                else:
                    raise ValueError(f"Invalid quorum set format: {q}") from e
            fbas.update_validator(pk, qset=qset_id)

    fbas.check_integrity()
    return fbas


@lru_cache(maxsize=128)
def compute_qset(fbas: FBASGraph, qset_vertex: str) -> QSet:
    """
    Recursively computes the QSet associated with the given qset vertex.
    """
    assert not fbas.is_validator(qset_vertex)
    threshold = fbas.threshold(qset_vertex)
    # validators are the children of the qset vertex that are validators:
    validators = frozenset(v for v in fbas.graph_view().successors(
        qset_vertex) if fbas.is_validator(v))
    # inner_qsets are the children of the qset vertex that are qset
    # vertices:
    inner_qsets = frozenset(compute_qset(fbas, q) for q in fbas.graph_view().successors(
        qset_vertex) if not fbas.is_validator(q))
    return QSet(threshold, validators, inner_qsets)


def qset_of(fbas: FBASGraph, n: str) -> Optional[QSet]:
    """
    Computes the QSet associated with the given vertex n based on the graph (does not use the qsets dict).
    n must be a validator vertex.
    """
    assert fbas.is_validator(n)
    # if n has no successors, then we don't know its qset:
    if fbas.graph_view().out_degree(n) == 0:
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

        for v in sorted(self.fbas.get_validators()):  # Sort for consistent output
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
