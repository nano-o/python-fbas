"""
Tests for FBASGraph equality comparison.
"""

import json
import tempfile
from pathlib import Path
import pytest
from python_fbas.fbas_graph import FBASGraph
from python_fbas.serialization import serialize, deserialize


class TestFBASGraphEquality:
    """Test FBASGraph equality comparison."""

    def test_simple_equality(self):
        """Test equality of two simple graphs."""
        # Create two identical graphs
        fbas1 = FBASGraph()
        fbas1.add_validator('v1', name='Validator One', homeDomain='example.com')
        fbas1.add_validator('v2', name='Validator Two')
        qset_id = fbas1.add_qset(threshold=1, components=['v2'])
        fbas1.update_validator('v1', qset=qset_id)
        
        fbas2 = FBASGraph()
        fbas2.add_validator('v1', name='Validator One', homeDomain='example.com')
        fbas2.add_validator('v2', name='Validator Two')
        qset_id = fbas2.add_qset(threshold=1, components=['v2'])
        fbas2.update_validator('v1', qset=qset_id)
        
        assert fbas1.equivalent(fbas2)
    
    def test_inequality_different_validators(self):
        """Test inequality when validators differ."""
        fbas1 = FBASGraph()
        fbas1.add_validator('v1')
        
        fbas2 = FBASGraph()
        fbas2.add_validator('v2')
        
        assert not fbas1.equivalent(fbas2)
    
    def test_inequality_different_structure(self):
        """Test inequality when graph structure differs."""
        fbas1 = FBASGraph()
        fbas1.add_validator('v1')
        fbas1.add_validator('v2')
        fbas1.add_validator('v3')
        qset_id = fbas1.add_qset(threshold=1, components=['v2', 'v3'])
        fbas1.update_validator('v1', qset=qset_id)
        
        fbas2 = FBASGraph()
        fbas2.add_validator('v1')
        fbas2.add_validator('v2')
        fbas2.add_validator('v3')
        qset_id = fbas2.add_qset(threshold=2, components=['v2', 'v3'])  # Different threshold
        fbas2.update_validator('v1', qset=qset_id)
        
        assert not fbas1.equivalent(fbas2)
    
    def test_serialization_roundtrip_small_files(self):
        """Test that graphs remain equal after serialization roundtrip for small test files."""
        test_data_dir = Path(__file__).parent / 'test_data' / 'small'
        
        # Get all JSON files in small directory
        json_files = list(test_data_dir.glob('*.json'))
        assert len(json_files) > 0, "No JSON files found in test_data/small"
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_content = f.read()
            
            # Load the original graph
            original = deserialize(json_content)
            
            # Serialize and deserialize
            serialized = serialize(original)
            roundtrip = deserialize(serialized)
            
            # Check equality
            assert original.equivalent(roundtrip), f"Roundtrip failed for {json_file.name}"
    
    def test_serialization_roundtrip_pubnet_files(self):
        """Test that graphs remain equal after serialization roundtrip for pubnet test files."""
        test_data_dir = Path(__file__).parent / 'test_data' / 'pubnet'
        
        # Get all JSON files in pubnet directory
        json_files = list(test_data_dir.glob('*.json'))
        assert len(json_files) > 0, "No JSON files found in test_data/pubnet"
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_content = f.read()
            
            # Load the original graph
            original = deserialize(json_content)
            
            # Serialize and deserialize
            serialized = serialize(original)
            roundtrip = deserialize(serialized)
            
            # Check equality
            assert original.equivalent(roundtrip), f"Roundtrip failed for {json_file.name}"
    
    def test_equality_with_complex_qsets(self):
        """Test equality with nested qsets."""
        fbas1 = FBASGraph()
        fbas1.add_validator('v1')
        fbas1.add_validator('v2')
        fbas1.add_validator('v3')
        
        # Create nested qsets
        inner_qset1 = fbas1.add_qset(threshold=1, components=['v2', 'v3'])
        outer_qset1 = fbas1.add_qset(threshold=2, components=['v1', inner_qset1])
        fbas1.update_validator('v1', qset=outer_qset1)
        
        fbas2 = FBASGraph()
        fbas2.add_validator('v1')
        fbas2.add_validator('v2')
        fbas2.add_validator('v3')
        
        # Create same nested qsets
        inner_qset2 = fbas2.add_qset(threshold=1, components=['v2', 'v3'])
        outer_qset2 = fbas2.add_qset(threshold=2, components=['v1', inner_qset2])
        fbas2.update_validator('v1', qset=outer_qset2)
        
        assert fbas1.equivalent(fbas2)
