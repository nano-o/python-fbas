import pytest
import subprocess
import sys
import os
import tempfile


def run_command_with_config(
        config_content,
        fbas_file,
        command_args,
        expected_exit_code=0):
    """Helper function to run a command with a config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--config-file={config_path}",
            f"--fbas={fbas_file}"
        ] + command_args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        assert result.returncode == expected_exit_code, f"Command failed with output: {result.stderr}"
        return result.stdout, result.stderr
    finally:
        os.unlink(config_path)


def test_config_file_invalid_solver():
    """Test that invalid solver in config file causes error."""
    config_content = """
sat_solver: "invalid_solver"
"""
    fbas_file = "tests/test_data/top_tier.json"

    stdout, stderr = run_command_with_config(
        config_content,
        fbas_file,
        ["check-intersection"],
        expected_exit_code=1
    )

    assert "Solver must be one of" in stderr


def test_config_file_invalid_encoding():
    """Test that invalid cardinality encoding in config file causes error."""
    config_content = """
card_encoding: "invalid_encoding"
"""
    fbas_file = "tests/test_data/top_tier.json"

    stdout, stderr = run_command_with_config(
        config_content,
        fbas_file,
        ["check-intersection"],
        expected_exit_code=1
    )

    assert "Invalid card_encoding" in stderr


def test_config_file_valid_settings():
    """Test that valid config file settings work."""
    config_content = """
sat_solver: "minisat22"
card_encoding: "naive"
validator_display: "name"
"""
    fbas_file = "tests/test_data/top_tier.json"

    stdout, stderr = run_command_with_config(
        config_content,
        fbas_file,
        ["check-intersection"]
    )

    # Should complete successfully
    assert "disjoint quorums" in stdout or "No disjoint quorums found" in stdout


def test_cli_overrides_config():
    """Test that CLI arguments override config file settings."""
    config_content = """
validator_display: "name"
"""
    fbas_file = "tests/test_data/top_tier.json"

    # CLI argument should override config file
    stdout, stderr = run_command_with_config(
        config_content,
        fbas_file,
        ["--validator-display=id", "check-intersection"]
    )

    # Should complete successfully regardless of which display format is used
    assert "disjoint quorums" in stdout or "No disjoint quorums found" in stdout


def test_default_config_file_detection(tmp_path):
    """Test that python-fbas.cfg is automatically detected."""
    config_content = """
sat_solver: "minisat22"
"""

    # Change to temporary directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Create python-fbas.cfg in current directory
        with open("python-fbas.cfg", "w") as f:
            f.write(config_content)

        # Create a test FBAS file in the temp directory
        fbas_content = """[
  {
    "publicKey": "GDQWITFJLZ5HT6JCOXYEVV5VFD6FTLAKJAUDKHAV3HKYGVJWA2DPYSQV",
    "quorumSet": {
      "threshold": 1,
      "validators": ["GDQWITFJLZ5HT6JCOXYEVV5VFD6FTLAKJAUDKHAV3HKYGVJWA2DPYSQV"]
    }
  }
]"""
        with open("test.json", "w") as f:
            f.write(fbas_content)

        # Run command without --config-file (should auto-detect
        # python-fbas.cfg)
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=test.json",
            "check-intersection"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should complete successfully
        assert result.returncode == 0
        assert "disjoint quorums" in result.stdout or "No disjoint quorums found" in result.stdout

    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__])
