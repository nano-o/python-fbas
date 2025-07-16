import pytest
import subprocess
import sys
import tempfile
import os
import yaml
from python_fbas.config import to_yaml, update


class TestShowConfig:
    def test_to_yaml_defaults(self):
        """Test YAML generation with default configuration."""
        # Reset to defaults
        update(
            stellar_data_url="https://radar.withobsrvr.com/api/v1/node",
            sat_solver="cryptominisat5",
            card_encoding="totalizer",
            max_sat_algo="LSU",
            validator_display="both",
            group_by=None,
            output=None
        )

        yaml_output = to_yaml()

        # Verify it contains expected sections
        assert "# python-fbas configuration file" in yaml_output
        assert "stellar_data_url: https://radar.withobsrvr.com/api/v1/node" in yaml_output
        assert "sat_solver: cryptominisat5" in yaml_output
        assert "card_encoding: totalizer" in yaml_output
        assert "max_sat_algo: LSU" in yaml_output
        assert "validator_display: both" in yaml_output

        # Should not contain None values
        assert "group_by:" not in yaml_output
        assert "output:" not in yaml_output

    def test_to_yaml_with_optional_values(self):
        """Test YAML generation with optional values set."""
        update(
            group_by="homeDomain",
            output="problem.cnf"
        )

        yaml_output = to_yaml()

        # Should contain optional values
        assert "group_by: homeDomain" in yaml_output
        assert "output: problem.cnf" in yaml_output

    def test_to_yaml_is_valid_yaml(self):
        """Test that generated YAML is parseable."""
        yaml_output = to_yaml()

        # Extract just the non-comment lines for parsing
        lines = yaml_output.split('\n')
        config_lines = [
            line for line in lines if not line.strip().startswith('#') and line.strip()]
        config_yaml = '\n'.join(config_lines)

        # Should parse without errors
        parsed = yaml.safe_load(config_yaml)
        assert isinstance(parsed, dict)
        assert "stellar_data_url" in parsed

    def test_show_config_cli_command(self):
        """Test show-config CLI command."""
        cmd = [sys.executable, "-m", "python_fbas.main", "show-config"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "# python-fbas configuration file" in result.stdout
        assert "stellar_data_url:" in result.stdout
        assert "sat_solver:" in result.stdout

    def test_show_config_with_config_file(self):
        """Test show-config with existing config file."""
        config_content = """
sat_solver: "glucose30"
validator_display: "id"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            cmd = [
                sys.executable,
                "-m",
                "python_fbas.main",
                f"--config-file={config_path}",
                "show-config"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0
            assert "sat_solver: glucose30" in result.stdout
            assert "validator_display: id" in result.stdout
        finally:
            os.unlink(config_path)

    def test_show_config_with_cli_override(self):
        """Test that CLI arguments override config in show-config output."""
        config_content = """
sat_solver: "minisat22"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            cmd = [
                sys.executable,
                "-m",
                "python_fbas.main",
                f"--config-file={config_path}",
                "--sat-solver=glucose41",
                "show-config"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0
            # CLI should override config file
            assert "sat_solver: glucose41" in result.stdout
            assert "sat_solver: minisat22" not in result.stdout
        finally:
            os.unlink(config_path)

    def test_show_config_no_fbas_required(self):
        """Test that show-config doesn't require --fbas argument."""
        cmd = [sys.executable, "-m", "python_fbas.main", "show-config"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should succeed without --fbas
        assert result.returncode == 0
        assert "--fbas is required" not in result.stderr

    def test_show_config_piped_output_valid(self):
        """Test that piped output creates valid YAML config."""
        cmd = [sys.executable, "-m", "python_fbas.main", "show-config"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0

        # Extract non-comment lines and verify they parse as valid YAML
        lines = result.stdout.split('\n')
        config_lines = [
            line for line in lines if not line.strip().startswith('#') and line.strip()]
        config_yaml = '\n'.join(config_lines)

        parsed = yaml.safe_load(config_yaml)
        assert isinstance(parsed, dict)
        assert "stellar_data_url" in parsed
        assert "sat_solver" in parsed

    def test_show_config_roundtrip(self):
        """Test that show-config output can be used as input config."""
        # Generate config
        cmd1 = [sys.executable, "-m", "python_fbas.main", "show-config"]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(result1.stdout)
            config_path = f.name

        try:
            # Use generated config
            cmd2 = [
                sys.executable,
                "-m",
                "python_fbas.main",
                f"--config-file={config_path}",
                "show-config"
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True)

            assert result2.returncode == 0
            # Should produce equivalent output (may have slight formatting
            # differences)
            assert "stellar_data_url:" in result2.stdout
            assert "sat_solver:" in result2.stdout
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__])
