import pytest
import subprocess
import sys
import os
import json
from platformdirs import user_cache_dir


class TestPerUrlCaching:
    def _get_cache_filename(self, url: str) -> str:
        """Generate cache filename for URL (same logic as in pubnet_data.py)."""
        from urllib.parse import quote

        # Remove protocol for cleaner filenames
        url_without_protocol = url.replace(
            'https://',
            '').replace(
            'http://',
            '')

        # Use percent-encoding (RFC 3986) for cross-platform safety
        encoded = quote(url_without_protocol, safe='-._~')

        # Handle very long URLs by truncating if necessary
        max_length = 200
        if len(encoded) > max_length:
            if '/' in url_without_protocol:
                domain, path = url_without_protocol.split('/', 1)
                domain_encoded = quote(domain, safe='-._~')
                remaining_length = max_length - len(domain_encoded) - 1
                if remaining_length > 10:
                    path_encoded = quote(path, safe='-._~')[:remaining_length]
                    encoded = f"{domain_encoded}_{path_encoded}"
                else:
                    encoded = domain_encoded
            else:
                encoded = encoded[:max_length]

        return f"stellar_network_data_{encoded}.json"

    def _get_cache_path(self, url: str) -> str:
        """Get full path to cache file for URL."""
        cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
        filename = self._get_cache_filename(url)
        return os.path.join(cache_dir, filename)

    def test_different_urls_create_separate_cache_files(self):
        """Test that different URLs create separate cache files."""
        # Test with default URL first
        cmd1 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "update-cache"
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0, f"Default update-cache failed: {result1.stderr}"

        # Test with different URL
        cmd2 = [
            sys.executable,
            "-m",
            "python_fbas.main",
            "--fbas=https://api.stellaratlas.io/v1/node",
            "update-cache"
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0, f"StellarAtlas update-cache failed: {result2.stderr}"

        # Check that both cache files exist
        default_url = "https://radar.withobsrvr.com/api/v1/node"  # from config
        stellaratlas_url = "https://api.stellaratlas.io/v1/node"

        default_cache_path = self._get_cache_path(default_url)
        stellaratlas_cache_path = self._get_cache_path(stellaratlas_url)

        # Debug: check what files actually exist
        cache_dir = user_cache_dir('python-fbas', 'SDF')
        actual_files = os.listdir(
            cache_dir) if os.path.exists(cache_dir) else []

        assert os.path.exists(
            default_cache_path), f"Default cache file should exist at {default_cache_path}. Actual files: {actual_files}"
        assert os.path.exists(
            stellaratlas_cache_path), f"StellarAtlas cache file should exist at {stellaratlas_cache_path}. Actual files: {actual_files}"

        # Verify they have different filenames
        assert default_cache_path != stellaratlas_cache_path

    def test_cache_persistence_across_different_urls(self):
        """Test that cache for one URL persists when using another URL."""
        default_url = "https://radar.withobsrvr.com/api/v1/node"

        # Update cache for default URL
        cmd1 = [
            sys.executable,
            "-m",
            "python_fbas.main",
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
            "--fbas=https://api.stellaratlas.io/v1/node",
            "update-cache"
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0

        # Verify default cache still exists and wasn't modified
        assert os.path.exists(default_cache_path)
        default_mtime_after = os.path.getmtime(default_cache_path)
        assert default_mtime_before == default_mtime_after, "Default cache should not be modified when updating different URL"

    def test_cached_data_contains_correct_url(self):
        """Test that cached data contains the correct source URL."""
        url = "https://api.stellaratlas.io/v1/node"

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

    def test_cached_data_used_message_includes_url(self):
        """Test that cache usage messages include the specific URL."""
        url = "https://api.stellaratlas.io/v1/node"

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

    def test_update_cache_shows_specific_url_and_file(self):
        """Test that update-cache shows both URL and cache file path."""
        url = "https://api.stellaratlas.io/v1/node"

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
