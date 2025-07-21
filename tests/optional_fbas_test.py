import pytest
import subprocess
import sys


class TestOptionalFbas:
    def test_fbas_optional_with_defaults(self):
        """Test that commands work without --fbas using defaults."""
        # This test may fail if network is down or URL is unreachable,
        # but that's expected behavior
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "show-config"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "stellar_data_url:" in result.stdout

    def test_fbas_explicit_file_still_works(self):
        """Test that explicit --fbas with file path still works."""
        fbas_file = "tests/test_data/small/top_tier.json"

        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            f"--fbas={fbas_file}",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert f"Using local FBAS file: {fbas_file}" in result.stderr
        assert "disjoint quorums" in result.stdout or "No disjoint quorums found" in result.stdout

    def test_fbas_explicit_url_still_works(self):
        """Test that explicit --fbas with URL still works."""
        url = "https://radar.withobsrvr.com/api/v1/node"

        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            f"--fbas={url}",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert f"Using Stellar network data from: {url}" in result.stderr
        assert ("Cache:" in result.stdout or "Updating cache" in result.stdout)

    def test_data_source_displayed(self):
        """Test that data source information is displayed."""
        fbas_file = "tests/test_data/small/top_tier.json"

        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            f"--fbas={fbas_file}",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "Using local FBAS file:" in result.stderr

    def test_update_cache_validation(self):
        """Test that --update-cache validation works with default URL."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            "--update-cache",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should work since default is a URL
        assert result.returncode == 0
        assert "Using default Stellar network data from:" in result.stderr

    def test_update_cache_with_file_fails(self):
        """Test that --update-cache fails when using a local file."""
        fbas_file = "tests/test_data/small/top_tier.json"

        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--fbas={fbas_file}",
            "--update-cache",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail because local files can't use cache update
        assert result.returncode == 1
        assert "Error: --update-cache can only be used with URLs" in result.stderr


if __name__ == '__main__':
    pytest.main([__file__])
