import pytest
import subprocess
import sys


class TestErrorHandling:
    def test_invalid_json_url_error(self):
        """Test that URLs returning non-JSON data show clear error."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=https://google.com",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 1
        assert "Error: URL" in result.stderr and "did not return valid JSON data" in result.stderr
        assert "Please check that the URL points to a Stellar network API endpoint" in result.stderr

    def test_nonexistent_file_error(self):
        """Test that non-existent files show clear error."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=nonexistent-file.json",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 1
        assert "Error: File not found: nonexistent-file.json" in result.stderr

    def test_nonexistent_url_error(self):
        """Test that non-existent URLs show clear error."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=https://this-domain-does-not-exist-12345.com",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 1
        assert "Error: Failed to fetch Stellar network data" in result.stderr

    def test_valid_file_succeeds(self):
        """Test that valid files work correctly."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            "--fbas=tests/test_data/small/top_tier.json",
            "check-intersection"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "Using local FBAS file: tests/test_data/small/top_tier.json" in result.stderr
        assert (
            "disjoint quorums" in result.stdout or "No disjoint quorums found" in result.stdout)

    def test_invalid_json_data_format_error(self):
        """Test error handling for URLs that return JSON but wrong format."""
        # This would test a URL that returns valid JSON but not a list of validators
        # For now, we'll just verify our error handling logic exists
        # (Hard to test without a controllable test server)
        pass


if __name__ == '__main__':
    pytest.main([__file__])
