import pytest
import subprocess
import sys
import os
import json
from platformdirs import user_cache_dir

from python_fbas.pubnet_data import _get_cache_filename


@pytest.mark.network
class TestPerUrlCaching:
    def _get_cache_path(self, url: str) -> str:
        """Get full path to cache file for URL."""
        cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
        filename = _get_cache_filename(url)
        return os.path.join(cache_dir, filename)

    def test_different_urls_create_separate_cache_files(self, network_url, network_alt_url, network_config_file):
        """Test that different URLs create separate cache files."""
        # Test with default URL first
        cmd1 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--config-file={network_config_file}",
            "update-cache"
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0, f"Default update-cache failed: {result1.stderr}"

        # Test with different URL
        cmd2 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--fbas={network_alt_url}",
            "update-cache"
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0, f"Alternate update-cache failed: {result2.stderr}"

        # Check that both cache files exist
        default_url = network_url
        alt_url = network_alt_url

        default_cache_path = self._get_cache_path(default_url)
        alt_cache_path = self._get_cache_path(alt_url)

        # Debug: check what files actually exist
        cache_dir = user_cache_dir('python-fbas', 'SDF')
        actual_files = os.listdir(
            cache_dir) if os.path.exists(cache_dir) else []

        assert os.path.exists(
            default_cache_path), f"Default cache file should exist at {default_cache_path}. Actual files: {actual_files}"
        assert os.path.exists(
            alt_cache_path), f"Alternate cache file should exist at {alt_cache_path}. Actual files: {actual_files}"

        # Verify they have different filenames
        assert default_cache_path != alt_cache_path

    def test_cache_persistence_across_different_urls(self, network_url, network_alt_url, network_config_file):
        """Test that cache for one URL persists when using another URL."""
        default_url = network_url

        # Update cache for default URL
        cmd1 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--config-file={network_config_file}",
            "update-cache"
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0

        # Get timestamp of default cache
        default_cache_path = self._get_cache_path(default_url)
        default_mtime_before = os.path.getmtime(default_cache_path)

        # Update cache for different URL
        cmd2 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--fbas={network_alt_url}",
            "update-cache"
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0

        # Verify default cache still exists and wasn't modified
        assert os.path.exists(default_cache_path)
        default_mtime_after = os.path.getmtime(default_cache_path)
        assert default_mtime_before == default_mtime_after, "Default cache should not be modified when updating different URL"

    def test_cached_data_contains_correct_url(self, network_alt_url):
        """Test that cached data contains the correct source URL."""
        url = network_alt_url

        # Update cache for specific URL
        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--fbas={url}",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Read cache file and verify URL
        cache_path = self._get_cache_path(url)
        assert os.path.exists(cache_path)

        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        assert cache_data['source_url'] == url
        assert 'cached_at' in cache_data
        assert 'validators' in cache_data
        assert isinstance(cache_data['validators'], list)

    def test_cached_data_used_message_includes_url(self, network_alt_url):
        """Test that cache usage messages include the specific URL."""
        url = network_alt_url

        # First update cache
        cmd1 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--fbas={url}",
            "update-cache"
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0

        # Then use the cache
        cmd2 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            f"--fbas={url}",
            "check-intersection"
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0

        # Verify cache usage message includes URL
        assert f"Cache: Using cached data for {url}" in result2.stdout

    def test_update_cache_shows_specific_url_and_file(self, network_alt_url):
        """Test that update-cache shows both URL and cache file path."""
        url = network_alt_url

        cmd = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--log-level=INFO",
            f"--fbas={url}",
            "update-cache"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert f"Updating cache for URL: {url}" in result.stdout
        assert "Cache file:" in result.stdout
        assert f"Successfully updated cache for: {url}" in result.stderr


if __name__ == '__main__':
    pytest.main([__file__])
