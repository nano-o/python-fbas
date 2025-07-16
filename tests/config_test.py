import pytest
import tempfile
import os
from python_fbas.config import load_from_file, load_config_file, get, update


class TestConfigFile:
    def test_load_valid_config(self):
        """Test loading a valid YAML config file."""
        config_content = """
stellar_data_url: "https://custom.example.com/api"
sat_solver: "minisat22"
card_encoding: "naive"
max_sat_algo: "RC2"
validator_display: "name"
group_by: "homeDomain"
output: "test.cnf"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_data = load_from_file(config_path)
            expected = {
                'stellar_data_url': 'https://custom.example.com/api',
                'sat_solver': 'minisat22',
                'card_encoding': 'naive',
                'max_sat_algo': 'RC2',
                'validator_display': 'name',
                'group_by': 'homeDomain',
                'output': 'test.cnf'
            }
            assert config_data == expected
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_from_file("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_from_file(config_path)
        finally:
            os.unlink(config_path)

    def test_load_config_file_explicit_path(self):
        """Test load_config_file with explicit path."""
        config_content = """
sat_solver: "glucose30"
card_encoding: "totalizer"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Reset to defaults
            update(sat_solver='cryptominisat5', card_encoding='totalizer')

            load_config_file(config_path)
            cfg = get()
            assert cfg.sat_solver == "glucose30"
            assert cfg.card_encoding == "totalizer"
        finally:
            os.unlink(config_path)

    def test_load_config_file_default_path(self, tmp_path):
        """Test load_config_file with default python-fbas.cfg in current directory."""
        config_content = """
validator_display: "id"
max_sat_algo: "RC2"
"""
        # Change to temporary directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Create python-fbas.cfg in current directory
            with open("python-fbas.cfg", "w") as f:
                f.write(config_content)

            # Reset to defaults
            update(validator_display='both', max_sat_algo='LSU')

            load_config_file()  # No explicit path
            cfg = get()
            assert cfg.validator_display == "id"
            assert cfg.max_sat_algo == "RC2"
        finally:
            os.chdir(original_cwd)

    def test_load_config_file_no_file(self, tmp_path):
        """Test load_config_file when no config file exists."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Reset to known state
            update(sat_solver='cryptominisat5')
            original_solver = get().sat_solver

            load_config_file()  # Should not change anything
            cfg = get()
            assert cfg.sat_solver == original_solver
        finally:
            os.chdir(original_cwd)

    def test_invalid_config_values(self):
        """Test validation of config values."""
        # Test invalid card_encoding
        config_content = """
card_encoding: "invalid_encoding"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid card_encoding"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

        # Test invalid max_sat_algo
        config_content = """
max_sat_algo: "INVALID"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid max_sat_algo"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

        # Test invalid validator_display
        config_content = """
validator_display: "invalid"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid validator_display"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

    def test_unknown_config_keys_ignored(self):
        """Test that unknown config keys are ignored with warning."""
        config_content = """
sat_solver: "minisat22"
unknown_key: "ignored_value"
another_unknown: 123
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Should not raise an error, unknown keys should be ignored
            load_config_file(config_path)
            cfg = get()
            assert cfg.sat_solver == "minisat22"
            # Unknown keys should not be present
            assert not hasattr(cfg, 'unknown_key')
        finally:
            os.unlink(config_path)

    def test_empty_config_file(self):
        """Test loading an empty config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            config_path = f.name

        try:
            # Should not raise an error
            load_config_file(config_path)
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__])
