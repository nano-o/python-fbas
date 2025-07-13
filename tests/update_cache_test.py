import pytest
import subprocess
import sys


class TestUpdateCache:
    def test_update_cache_default_url(self):
        """Test update-cache without --fbas uses default URL."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Updating cache for URL:" in result.stdout
        assert "Successfully updated cache for default URL:" in result.stdout

    def test_update_cache_with_url(self):
        """Test update-cache with --fbas URL."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=https://api.stellaratlas.io/v1/node",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Updating cache for URL: https://api.stellaratlas.io/v1/node" in result.stdout
        assert "Successfully updated cache for: https://api.stellaratlas.io/v1/node" in result.stdout

    def test_update_cache_with_file_errors(self):
        """Test that update-cache with local file shows clear error."""
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=tests/test_data/top_tier.json",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "Error: Cannot update cache for local files" in result.stderr
        assert "Cache updates only work with URLs" in result.stderr

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