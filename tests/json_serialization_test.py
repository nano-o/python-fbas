"""
Tests for JSON serialization and deserialization functionality.
"""

import json
import logging
import pytest
from pathlib import Path
from python_fbas.fbas_graph import FBASGraph
from python_fbas.serialization import FBASSerializer


class TestJSONSerialization:
    """Test JSON serialization and deserialization methods."""

    def test_to_json_basic(self):
        """Test basic serialization to python-fbas format."""
        fbas = FBASGraph()
        fbas.add_validator('v1')
        fbas.add_validator('v2')

        # Add attributes
        fbas.graph.nodes['v1']['name'] = 'Validator One'
        fbas.graph.nodes['v1']['homeDomain'] = 'example.com'

        # Add qset
        qset_id = fbas.add_qset(threshold=1, members=['v2'], qset_id='qset1')
        fbas.update_validator('v1', qset_id)

        serializer = FBASSerializer()
        json_str = serializer.serialize(fbas, format='python-fbas')

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert 'validators' in data
        assert 'qsets' in data
        assert len(data['validators']) == 2
        assert len(data['qsets']) == 1

    def test_from_json_python_fbas_basic(self):
        """Test basic deserialization from python-fbas format."""
        json_data = {
            "validators": [
                {
                    "id": "v1",
                    "qset": "_q1",
                    "attrs": {"name": "Validator One"}
                },
                {
                    "id": "v2",
                    "qset": None,
                    "attrs": {}
                }
            ],
            "qsets": {
                "_q1": {
                    "threshold": 1,
                    "members": ["v2"]
                }
            }
        }

        serializer = FBASSerializer()
        fbas = serializer.deserialize(json.dumps(json_data))

        assert len(fbas.validators) == 2
        assert 'v1' in fbas.validators
        assert 'v2' in fbas.validators
        assert fbas.vertice_attrs('v1')['name'] == 'Validator One'
        assert fbas.qset_of('v1').threshold == 1

    def test_round_trip_consistency(self):
        """Test that serialize -> deserialize produces identical graph."""
        # Create a more complex FBAS
        fbas_original = FBASGraph()
        for i in range(5):
            fbas_original.add_validator(f'v{i}')
            fbas_original.graph.nodes[f'v{i}']['name'] = f'Validator {i}'
            fbas_original.graph.nodes[f'v{i}']['homeDomain'] = f'domain{i}.com'

        # Create complex qsets with inner quorum sets
        inner_qset_id = fbas_original.add_qset(threshold=2, members=['v3', 'v4'], qset_id='inner_qset')
        main_qset_id = fbas_original.add_qset(threshold=2, members=['v1', 'inner_qset'], qset_id='main_qset')
        fbas_original.update_validator('v0', main_qset_id)

        simple_qset_id = fbas_original.add_qset(threshold=1, members=['v0'], qset_id='simple_qset')
        fbas_original.update_validator('v2', simple_qset_id)

        # Round trip
        serializer = FBASSerializer()
        json_str = serializer.serialize(fbas_original, format='python-fbas')
        fbas_restored = serializer.deserialize(json_str)

        # Verify consistency
        assert fbas_original.validators == fbas_restored.validators

        # Check qsets match
        for v in fbas_original.validators:
            original_qset = fbas_original.qset_of(v)
            restored_qset = fbas_restored.qset_of(v)
            assert original_qset == restored_qset

        # Check attributes match
        for v in fbas_original.validators:
            orig_attrs = fbas_original.vertice_attrs(v)
            rest_attrs = fbas_restored.vertice_attrs(v)
            assert orig_attrs == rest_attrs


class TestFormatDetection:
    """Test automatic format detection."""

    def test_detect_stellarbeat_format(self):
        """Test detection of stellarbeat format."""
        stellarbeat_data = [
            {
                "publicKey": "PK1",
                "quorumSet": {
                    "threshold": 1,
                    "validators": ["PK2"],
                    "innerQuorumSets": []
                }
            }
        ]

        serializer = FBASSerializer()
        assert serializer.detect_format(stellarbeat_data) == 'stellarbeat'

    def test_detect_python_fbas_format(self):
        """Test detection of python-fbas format."""
        python_fbas_data = {
            "validators": [{"id": "v1", "qset": None, "attrs": {}}],
            "qsets": {}
        }

        serializer = FBASSerializer()
        assert serializer.detect_format(python_fbas_data) == 'python-fbas'

    def test_detect_unknown_format(self):
        """Test detection of unknown formats."""
        # String list (like organization names)
        serializer = FBASSerializer()
        assert serializer.detect_format(["org1", "org2"]) == 'unknown'

        # Random dict
        serializer = FBASSerializer()
        assert serializer.detect_format({"random": "data"}) == 'unknown'

        # Empty list
        serializer = FBASSerializer()
        assert serializer.detect_format(
            []) == 'stellarbeat'  # Empty lists default to stellarbeat

    def test_from_json_auto_detection(self, caplog):
        """Test that from_json auto-detects formats and logs them."""
        with caplog.at_level(logging.INFO):
            # Test stellarbeat
            stellarbeat_data = [
                {"publicKey": "PK1", "quorumSet": {"threshold": 1,
                                                   "validators": [], "innerQuorumSets": []}}
            ]
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps(stellarbeat_data))

            # Test python-fbas
            python_fbas_data = {
                "validators": [{"id": "v1", "qset": None, "attrs": {}}],
                "qsets": {}
            }
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps(python_fbas_data))

        # Check that format detection was logged
        log_messages = [record.message for record in caplog.records]
        assert any(
            "Detected JSON format: stellarbeat" in msg for msg in log_messages)
        assert any(
            "Detected JSON format: python-fbas" in msg for msg in log_messages)

    def test_from_json_string_input(self):
        """Test that from_json accepts JSON strings."""
        json_str = json.dumps({
            "validators": [{"id": "v1", "qset": None, "attrs": {}}],
            "qsets": {}
        })

        serializer = FBASSerializer()
        fbas = serializer.deserialize(json_str)
        assert len(fbas.validators) == 1
        assert 'v1' in fbas.validators


class TestFileBasedSerialization:
    """Test serialization with actual test files."""

    @pytest.fixture
    def test_data_paths(self):
        """Get paths to stellarbeat and python-fbas test files."""
        test_data_dir = Path(__file__).parent / 'test_data'

        # Stellarbeat format files (only pubnet_obsrvr.json is still in stellarbeat format)
        stellarbeat_files = [
            f for f in [test_data_dir / 'pubnet' / 'pubnet_obsrvr.json']
            if f.exists()
        ]

        # Python-fbas format files from all subdirectories
        python_fbas_files = []
        for subdir in ['small', 'pubnet', 'random']:
            subdir_path = test_data_dir / subdir
            if subdir_path.exists():
                python_fbas_files.extend(list(subdir_path.glob('*.json')))
        
        # Remove stellarbeat files and non-FBAS files from python-fbas list
        python_fbas_files = [
            f for f in python_fbas_files 
            if f.name not in ['pubnet_obsrvr.json', 'validators_broken_1.json', 'top_tier_orgs.json']
            and not f.name.endswith('_orgs.json')  # Exclude organization files
            and not f.name.endswith('_orgs_orgs.json')  # Exclude organization files
        ]

        return stellarbeat_files, python_fbas_files

    def test_load_stellarbeat_files(self, test_data_paths):
        """Test loading all stellarbeat test files."""
        stellarbeat_files, _ = test_data_paths

        for file_path in stellarbeat_files:
            with open(file_path) as f:
                data = json.load(f)

            try:
                serializer = FBASSerializer()
                fbas = serializer.deserialize(json.dumps(data))
                assert fbas.validators is not None
                # Basic integrity check
                fbas.check_integrity()
            except Exception as e:
                pytest.fail(f"Failed to load {file_path.name}: {e}")

    def test_load_python_fbas_files(self, test_data_paths):
        """Test loading all python-fbas test files."""
        _, python_fbas_files = test_data_paths

        for file_path in python_fbas_files:
            with open(file_path) as f:
                json_str = f.read()

            try:
                serializer = FBASSerializer()
                fbas = serializer.deserialize(json_str)
                assert fbas.validators is not None
                # Basic integrity check
                fbas.check_integrity()
            except Exception as e:
                pytest.fail(f"Failed to load {file_path.name}: {e}")

    def test_stellarbeat_to_python_fbas_consistency(self, test_data_paths):
        """Test that converting stellarbeat to python-fbas maintains consistency."""
        stellarbeat_files, _ = test_data_paths

        # Skip if no stellarbeat files available
        if not stellarbeat_files:
            pytest.skip("No stellarbeat format files available for testing")

        # Test available files
        test_files = stellarbeat_files

        for file_path in test_files:
            with open(file_path) as f:
                stellarbeat_data = json.load(f)

            # Load as stellarbeat
            serializer = FBASSerializer()
            fbas_stellarbeat = serializer.deserialize(json.dumps(stellarbeat_data))

            # Convert to python-fbas and reload
            python_fbas_json = serializer.serialize(fbas_stellarbeat, format='python-fbas')
            fbas_python_fbas = serializer.deserialize(python_fbas_json)

            # Should be equivalent
            assert fbas_stellarbeat.validators == fbas_python_fbas.validators

            # Check qsets are equivalent
            for v in fbas_stellarbeat.validators:
                qset1 = fbas_stellarbeat.qset_of(v)
                qset2 = fbas_python_fbas.qset_of(v)
                assert qset1 == qset2


class TestUUIDGeneration:
    """Test UUID-based qset ID generation."""

    def test_add_qset_generates_uuid(self):
        """Test that add_qset generates UUID-based IDs."""
        fbas = FBASGraph()
        fbas.add_validator('v1')
        fbas.add_validator('v2')
        fbas.add_validator('v3')
        fbas.add_validator('v4')
        fbas.add_validator('v5')

        # Add multiple different qsets to ensure unique UUIDs
        qset_ids = set()
        for i in range(5):
            # Create different qsets by varying validator combinations
            # [v1], [v1,v2], etc.
            validators = [f'v{j+1}' for j in range(i + 1)]
            qset_id = fbas.add_qset(threshold=len(validators), members=validators)
            assert qset_id.startswith('_q')
            assert len(qset_id) == 34  # _q + 32 hex chars
            qset_ids.add(qset_id)

        # All IDs should be unique
        assert len(qset_ids) == 5

    def test_manual_qset_ids_preserved(self):
        """Test that manually specified qset IDs are preserved."""
        json_data = {
            "validators": [
                {"id": "v1", "qset": "myCustomQset", "attrs": {}},
                {"id": "v2", "qset": "org-qset-1", "attrs": {}},
                # Old format still works
                {"id": "v3", "qset": "_q1", "attrs": {}},
            ],
            "qsets": {
                "myCustomQset": {"threshold": 1, "members": ["v2"]},
                "org-qset-1": {"threshold": 2, "members": ["v1", "v3"]},
                "_q1": {"threshold": 1, "members": ["v1"]}
            }
        }

        serializer = FBASSerializer()
        fbas = serializer.deserialize(json.dumps(json_data))

        # Check that custom IDs are preserved
        assert "myCustomQset" in fbas.graph.nodes
        assert "org-qset-1" in fbas.graph.nodes
        assert "_q1" in fbas.graph.nodes

        # Verify connections
        assert list(fbas.graph.successors('v1'))[0] == "myCustomQset"
        assert list(fbas.graph.successors('v2'))[0] == "org-qset-1"
        assert list(fbas.graph.successors('v3'))[0] == "_q1"

    def test_duplicate_id_detection(self):
        """Test that duplicate IDs are detected and reported."""
        # Test duplicate validator IDs
        json_data = {
            "validators": [
                {"id": "v1", "qset": None, "attrs": {}},
                {"id": "v1", "qset": None, "attrs": {}}  # Duplicate
            ],
            "qsets": {}
        }

        with pytest.raises(ValueError, match="Duplicate IDs found: \\['v1'\\]"):
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps(json_data))

        # Test validator-qset collision
        json_data = {
            "validators": [
                {"id": "shared_id", "qset": None, "attrs": {}}
            ],
            "qsets": {
                # Same ID as validator
                "shared_id": {"threshold": 1, "members": []}
            }
        }

        with pytest.raises(ValueError, match="Duplicate IDs found: \\['shared_id'\\]"):
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps(json_data))

        # Test multiple duplicates
        json_data = {
            "validators": [
                {"id": "v1", "qset": None, "attrs": {}},
                {"id": "v2", "qset": None, "attrs": {}},
                {"id": "v1", "qset": None, "attrs": {}}  # Duplicate v1
            ],
            "qsets": {
                "v2": {"threshold": 1, "members": []},  # Duplicate v2
            }
        }

        with pytest.raises(ValueError, match="Duplicate IDs found: \\['v1', 'v2'\\]"):
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps(json_data))

    def test_round_trip_with_uuid_qsets(self):
        """Test that UUID-generated qsets survive round-trip serialization."""
        fbas1 = FBASGraph()
        fbas1.add_validator('v1')
        fbas1.add_validator('v2')
        fbas1.add_validator('v3')

        # Create qsets using add_qset (will generate UUIDs)
        qset1_id = fbas1.add_qset(threshold=2, members=['v2', 'v3'])
        fbas1.update_validator('v1', qset1_id)

        # Get the generated qset ID
        qset_id = list(fbas1.graph.successors('v1'))[0]
        assert qset_id.startswith('_q')
        assert len(qset_id) == 34

        # Round trip
        serializer = FBASSerializer()
        json_str = serializer.serialize(fbas1, format='python-fbas')
        fbas2 = serializer.deserialize(json_str)

        # Check that the UUID-based ID is preserved
        qset_id2 = list(fbas2.graph.successors('v1'))[0]
        assert qset_id == qset_id2

        # Check that qsets are equivalent
        assert fbas1.qset_of('v1') == fbas2.qset_of('v1')


class TestStellarBeatSerialization:
    """Test serialization to stellarbeat format."""

    def test_to_json_stellarbeat_basic(self):
        """Test basic serialization to stellarbeat format."""
        fbas = FBASGraph()
        fbas.add_validator('V1')
        fbas.add_validator('V2')
        fbas.add_validator('V3')

        # Add attributes
        fbas.graph.nodes['V1']['name'] = 'Validator One'
        fbas.graph.nodes['V1']['homeDomain'] = 'example.com'

        # Add qset
        qset_id = fbas.add_qset(threshold=2, members=['V2', 'V3'], qset_id='qset1')
        fbas.update_validator('V1', qset_id)

        # Serialize to stellarbeat format
        serializer = FBASSerializer()
        json_str = serializer.serialize(fbas, format='stellarbeat')
        data = json.loads(json_str)

        # Verify structure
        assert isinstance(data, list)
        assert len(data) == 3

        # Find V1 in the output
        v1_data = next(v for v in data if v['publicKey'] == 'V1')
        assert v1_data['name'] == 'Validator One'
        assert v1_data['homeDomain'] == 'example.com'
        assert 'quorumSet' in v1_data
        assert v1_data['quorumSet']['threshold'] == 2
        assert set(v1_data['quorumSet']['validators']) == {'V2', 'V3'}
        assert v1_data['quorumSet']['innerQuorumSets'] == []

        # V2 and V3 should not have quorumSets
        v2_data = next(v for v in data if v['publicKey'] == 'V2')
        assert 'quorumSet' not in v2_data

    def test_stellarbeat_round_trip(self):
        """Test round-trip: stellarbeat -> FBASGraph -> stellarbeat."""
        original_data = [
            {
                "publicKey": "PK1",
                "name": "Validator 1",
                "quorumSet": {
                    "threshold": 2,
                    "validators": ["PK2", "PK3"],
                    "innerQuorumSets": [
                        {
                            "threshold": 1,
                            "validators": ["PK4"],
                            "innerQuorumSets": []
                        }
                    ]
                }
            },
            {
                "publicKey": "PK2",
                "name": "Validator 2"
            },
            {
                "publicKey": "PK3",
                "name": "Validator 3"
            },
            {
                "publicKey": "PK4",
                "name": "Validator 4"
            }
        ]

        # Load from stellarbeat format
        serializer = FBASSerializer()
        fbas = serializer.deserialize(json.dumps(original_data))

        # Serialize back to stellarbeat format
        json_str = serializer.serialize(fbas, format='stellarbeat')
        restored_data = json.loads(json_str)

        # Should have the same validators
        assert len(restored_data) == 4

        # Check PK1's quorum set is preserved
        pk1_restored = next(
            v for v in restored_data if v['publicKey'] == 'PK1')
        pk1_original = original_data[0]

        assert pk1_restored['name'] == pk1_original['name']
        assert pk1_restored['quorumSet']['threshold'] == pk1_original['quorumSet']['threshold']
        assert set(
            pk1_restored['quorumSet']['validators']) == set(
            pk1_original['quorumSet']['validators'])
        assert len(pk1_restored['quorumSet']['innerQuorumSets']) == 1

        inner_qset = pk1_restored['quorumSet']['innerQuorumSets'][0]
        assert inner_qset['threshold'] == 1
        assert inner_qset['validators'] == ['PK4']

    def test_full_round_trip_stellarbeat_python_fbas_stellarbeat(self):
        """Test full round-trip: stellarbeat -> python-fbas -> stellarbeat."""
        # Use a more complex example
        original_data = [
            {
                "publicKey": "V1",
                "name": "Validator 1",
                "homeDomain": "example1.com",
                "active": True,
                "quorumSet": {
                    "threshold": 3,
                    "validators": ["V2"],
                    "innerQuorumSets": [
                        {
                            "threshold": 2,
                            "validators": ["V3", "V4"],
                            "innerQuorumSets": []
                        },
                        {
                            "threshold": 1,
                            "validators": [],
                            "innerQuorumSets": [
                                {
                                    "threshold": 1,
                                    "validators": ["V5"],
                                    "innerQuorumSets": []
                                }
                            ]
                        }
                    ]
                }
            },
            {"publicKey": "V2", "name": "Validator 2", "active": False},
            {"publicKey": "V3", "name": "Validator 3"},
            {"publicKey": "V4", "name": "Validator 4"},
            {"publicKey": "V5", "name": "Validator 5"}
        ]

        # Step 1: Load from stellarbeat format
        serializer = FBASSerializer()
        fbas1 = serializer.deserialize(json.dumps(original_data))

        # Step 2: Convert to python-fbas format
        python_fbas_json = serializer.serialize(fbas1, format='python-fbas')

        # Step 3: Load from python-fbas format
        fbas2 = serializer.deserialize(python_fbas_json)

        # Step 4: Convert back to stellarbeat format
        final_json = serializer.serialize(fbas2, format='stellarbeat')
        final_data = json.loads(final_json)

        # Verify we have all validators
        assert len(final_data) == 5

        # Check that complex quorum set structure is preserved
        v1_final = next(v for v in final_data if v['publicKey'] == 'V1')
        v1_original = original_data[0]

        # Check attributes
        assert v1_final['name'] == v1_original['name']
        assert v1_final['homeDomain'] == v1_original['homeDomain']
        assert v1_final['active'] == v1_original['active']

        # Check quorum set structure
        assert v1_final['quorumSet']['threshold'] == 3
        assert v1_final['quorumSet']['validators'] == ['V2']
        assert len(v1_final['quorumSet']['innerQuorumSets']) == 2

        # Check inner qsets (order-independent)
        inner_qsets = v1_final['quorumSet']['innerQuorumSets']
        
        # Find the qset with threshold 2
        threshold_2_qset = next(q for q in inner_qsets if q['threshold'] == 2)
        assert set(threshold_2_qset['validators']) == {'V3', 'V4'}
        assert threshold_2_qset['innerQuorumSets'] == []

        # Find the qset with threshold 1 (nested structure)
        threshold_1_qset = next(q for q in inner_qsets if q['threshold'] == 1)
        assert threshold_1_qset['validators'] == []
        assert len(threshold_1_qset['innerQuorumSets']) == 1
        assert threshold_1_qset['innerQuorumSets'][0]['validators'] == ['V5']


class TestPubnetRoundTrip:
    """Test round-trip serialization with real pubnet data."""

    def test_pubnet_round_trip_consistency(self):
        """Test that pubnet data survives round-trip serialization."""
        # Load pubnet data from the downloaded file
        pubnet_file = Path(__file__).parent / \
            'test_data' / 'pubnet' / 'pubnet_obsrvr.json'

        if not pubnet_file.exists():
            pytest.skip("Pubnet data file not found")

        with open(pubnet_file) as f:
            stellarbeat_data = json.load(f)

        # Load as stellarbeat format
        serializer = FBASSerializer()
        fbas_original = serializer.deserialize(json.dumps(stellarbeat_data))

        # Basic sanity checks
        assert len(fbas_original.validators) > 0
        # Check we have some validators with quorum sets
        validators_with_qsets = sum(1 for v in fbas_original.validators if fbas_original.qset_of(v) is not None)
        assert validators_with_qsets > 0

        # Convert to python-fbas format
        python_fbas_json = serializer.serialize(fbas_original, format='python-fbas')

        # Load from python-fbas format
        fbas_restored = serializer.deserialize(python_fbas_json)

        # Verify the graphs are equivalent
        assert fbas_original.validators == fbas_restored.validators

        # Check that all validators have the same qsets
        validators_checked = 0
        for v in fbas_original.validators:
            qset_original = fbas_original.qset_of(v)
            qset_restored = fbas_restored.qset_of(v)

            # Both should be None or both should exist
            assert (qset_original is None) == (qset_restored is None)

            # If they exist, they should be equal
            if qset_original is not None:
                assert qset_original == qset_restored
                validators_checked += 1

        # Make sure we actually checked some validators with qsets
        assert validators_checked > 0

        # Check that all validator attributes are preserved
        for v in fbas_original.validators:
            orig_attrs = fbas_original.vertice_attrs(v).copy()
            rest_attrs = fbas_restored.vertice_attrs(v).copy()

            # Remove attributes that are not serialized
            for attr in ['quorumSet', 'quorumSetHashKey', 'threshold']:
                orig_attrs.pop(attr, None)
                rest_attrs.pop(attr, None)

            # Remove any internal attributes that might differ
            orig_attrs = {
                k: v for k,
                v in orig_attrs.items() if not k.startswith('_')}
            rest_attrs = {
                k: v for k,
                v in rest_attrs.items() if not k.startswith('_')}
            assert orig_attrs == rest_attrs

        # Verify graph structure is preserved
        assert fbas_original.graph.number_of_nodes(
        ) == fbas_restored.graph.number_of_nodes()
        assert fbas_original.graph.number_of_edges(
        ) == fbas_restored.graph.number_of_edges()

        # Both should pass integrity checks
        fbas_original.check_integrity()
        fbas_restored.check_integrity()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_self_referencing_qset_skipped(self):
        """Test that self-referencing qsets cause an error."""
        json_data = {
            "validators": [
                {"id": "v1", "qset": "q1", "attrs": {}},
            ],
            "qsets": {
                # q1 references itself
                "q1": {"threshold": 2, "members": ["v1", "q1"]},
            }
        }

        # An assert should fail here because self-referencing qsets are not allowed
        with pytest.raises(Exception):
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps(json_data))

    def test_empty_fbas_serialization(self):
        """Test serialization of empty FBAS."""
        fbas = FBASGraph()
        serializer = FBASSerializer()
        json_str = serializer.serialize(fbas, format='python-fbas')

        data = json.loads(json_str)
        assert data['validators'] == []
        assert data['qsets'] == {}

        # Should be able to deserialize
        fbas2 = serializer.deserialize(json_str)
        assert len(fbas2.validators) == 0

    def test_invalid_python_fbas_format(self):
        """Test error handling for invalid python-fbas format."""
        with pytest.raises(ValueError, match="Unknown or unsupported JSON format"):
            serializer = FBASSerializer()
            serializer.deserialize('["not", "a", "dict"]')

        with pytest.raises(ValueError, match="Unknown or unsupported JSON format"):
            serializer = FBASSerializer()
            serializer.deserialize('{"invalid": "format"}')

    def test_invalid_json_string(self):
        """Test error handling for invalid JSON strings."""
        with pytest.raises(json.JSONDecodeError):
            serializer = FBASSerializer()
            serializer.deserialize("invalid json")

    def test_unknown_format_error(self):
        """Test error when format cannot be determined."""
        with pytest.raises(ValueError, match="Unknown or unsupported JSON format"):
            serializer = FBASSerializer()
            serializer.deserialize(json.dumps({"random": "data"}))

    def test_validators_without_qsets(self):
        """Test handling validators without quorum sets."""
        json_data = {
            "validators": [
                {"id": "v1", "qset": None, "attrs": {"name": "Validator 1"}},
                {"id": "v2", "qset": None, "attrs": {}}
            ],
            "qsets": {}
        }

        serializer = FBASSerializer()
        fbas = serializer.deserialize(json.dumps(json_data))
        assert len(fbas.validators) == 2
        assert fbas.qset_of('v1') is None
        assert fbas.qset_of('v2') is None

    def test_complex_inner_quorum_sets(self):
        """Test complex nested inner quorum sets."""
        fbas = FBASGraph()
        fbas.add_validator('v1')
        fbas.add_validator('v2')
        fbas.add_validator('v3')
        fbas.add_validator('v4')

        # Create deeply nested qsets
        inner_inner_id = fbas.add_qset(threshold=1, members=['v4'], qset_id='inner_inner')
        inner_id = fbas.add_qset(threshold=1, members=['v3', 'inner_inner'], qset_id='inner')
        outer_id = fbas.add_qset(threshold=2, members=['v2', 'inner'], qset_id='outer')

        fbas.update_validator('v1', outer_id)

        # Round trip
        serializer = FBASSerializer()
        json_str = serializer.serialize(fbas, format='python-fbas')
        fbas2 = serializer.deserialize(json_str)

        assert fbas.qset_of('v1') == fbas2.qset_of('v1')
