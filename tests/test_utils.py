import os
import json
from pathlib import Path


def _get_test_data_file_path(name) -> str:
    """Get path to test data file, searching in subdirectories."""
    test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    
    # Check each subdirectory
    for subdir in ['small', 'pubnet', 'random']:
        file_path = os.path.join(test_data_dir, subdir, name)
        if os.path.exists(file_path):
            return file_path
    
    # Check root directory for files like top_tier_orgs.json
    root_path = os.path.join(test_data_dir, name)
    if os.path.exists(root_path):
        return root_path
    
    raise FileNotFoundError(f"Test data file '{name}' not found in any test_data subdirectory")


def get_validators_from_test_fbas(filename) -> list[dict]:
    path = _get_test_data_file_path(filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_fbas_from_test_file(filename):
    """Load an FBASGraph from a test data file.
    
    This is a convenience function that combines get_validators_from_test_fbas()
    and FBASSerializer.deserialize() to simplify the common pattern in tests.
    """
    from python_fbas.serialization import deserialize
    json_data = json.dumps(get_validators_from_test_fbas(filename))
    return deserialize(json_data)


def get_test_data_list(dirs: list[str] = ['small', 'pubnet']) -> dict[str, list[dict]]:
    test_data_dir = Path(__file__).parent / 'test_data'
    data = {}

    # Search in all subdirectories
    for subdir in dirs:
        subdir_path = test_data_dir / subdir
        if subdir_path.exists():
            for file in subdir_path.glob('*.json'):
                # Load and store with filename as key
                with open(file, 'r', encoding='utf-8') as f:
                    data.update({file.name: json.load(f)})

    return data
