import pytest
import subprocess
import sys


class TestUpdateCache:
    @pytest.mark.network
    def test_update_cache_default_url(self, network_url, network_config_file):
        """Test update-cache without --fbas uses default URL."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            f"--config-file={network_config_file}",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert f"Updating cache for URL: {network_url}" in result.stdout
        assert f"Successfully updated cache for default URL: {network_url}" in result.stderr

    @pytest.mark.network
    def test_update_cache_with_url(self, network_alt_url):
        """Test update-cache with --fbas URL."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            f"--fbas={network_alt_url}",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert f"Updating cache for URL: {network_alt_url}" in result.stdout
        assert f"Successfully updated cache for: {network_alt_url}" in result.stderr

    def test_update_cache_with_file_errors(self):
        """Test that update-cache with local file shows clear error."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=tests/test_data/small/top_tier.json",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 1
        assert "Error: Cannot update cache for local files" in result.stderr
        assert "Cache updates only work with URLs" in result.stderr

    @pytest.mark.network
    def test_update_cache_with_invalid_url(self):
        """Test update-cache with invalid URL shows appropriate error."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=https://invalid-domain-12345.com",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with proper error handling
        assert result.returncode == 1
        assert "Updating cache for URL: https://invalid-domain-12345.com" in result.stdout
        assert "Failed to update cache for URL: https://invalid-domain-12345.com" in result.stderr
        assert "Error:" in result.stderr

    @pytest.mark.network
    def test_update_cache_with_invalid_json_url(self):
        """Test update-cache with URL that returns invalid JSON."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=https://stellaratlas.io/api/v1/node",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with proper error handling
        assert result.returncode == 1
        assert "Updating cache for URL: https://stellaratlas.io/api/v1/node" in result.stdout
        assert "Failed to update cache for URL: https://stellaratlas.io/api/v1/node" in result.stderr
        assert "did not return valid JSON data" in result.stderr


if __name__ == '__main__':
    pytest.main([__file__])
