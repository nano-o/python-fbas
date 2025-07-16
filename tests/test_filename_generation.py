import pytest
from python_fbas.pubnet_data import _get_cache_filename


class TestFilenameGeneration:
    def test_basic_url_conversion(self):
        """Test basic URL to filename conversion."""
        url = "https://radar.withobsrvr.com/api/v1/node"
        expected = "stellar_network_data_radar.withobsrvr.com%2Fapi%2Fv1%2Fnode.json"
        assert _get_cache_filename(url) == expected

    def test_stellaratlas_url_conversion(self):
        """Test StellarAtlas URL conversion."""
        url = "https://api.stellaratlas.io/v1/node"
        expected = "stellar_network_data_api.stellaratlas.io%2Fv1%2Fnode.json"
        assert _get_cache_filename(url) == expected

    def test_url_with_port(self):
        """Test URL with port number."""
        url = "https://example.com:8080/api/v1/node"
        expected = "stellar_network_data_example.com%3A8080%2Fapi%2Fv1%2Fnode.json"
        assert _get_cache_filename(url) == expected

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        url = "https://example.com/api/v1/node?param1=value1&param2=value2"
        expected = "stellar_network_data_example.com%2Fapi%2Fv1%2Fnode%3Fparam1%3Dvalue1%26param2%3Dvalue2.json"
        assert _get_cache_filename(url) == expected

    def test_http_url(self):
        """Test HTTP (non-HTTPS) URL."""
        url = "http://example.com/api/v1/node"
        expected = "stellar_network_data_example.com%2Fapi%2Fv1%2Fnode.json"
        assert _get_cache_filename(url) == expected

    def test_long_url_truncation(self):
        """Test that very long URLs are truncated properly."""
        url = "https://very-long-domain-name-that-exceeds-normal-length.example.com/very/long/path/with/many/segments/that/should/be/truncated/to/reasonable/length/api/v1/node"
        result = _get_cache_filename(url)
        assert result.startswith("stellar_network_data_")
        assert result.endswith(".json")
        # Total length should be reasonable for all filesystems (prefix + 200 +
        # suffix)
        assert len(result) <= 250

    def test_special_characters(self):
        """Test URL with special characters gets properly encoded."""
        url = "https://example.com/path with spaces/file.json"
        result = _get_cache_filename(url)
        expected = "stellar_network_data_example.com%2Fpath%20with%20spaces%2Ffile.json.json"
        assert result == expected

    def test_international_domain(self):
        """Test international domain names are properly encoded."""
        url = "https://xn--d1acpjx3f.xn--p1ai/api/nodes"  # пример.рф in punycode
        result = _get_cache_filename(url)
        assert result.startswith("stellar_network_data_")
        assert result.endswith(".json")
        # Should contain the punycode domain
        assert "xn--d1acpjx3f.xn--p1ai" in result

    def test_all_safe_characters_preserved(self):
        """Test that safe characters are not encoded unnecessarily."""
        url = "https://example-site.com/api_v1/nodes.json"
        result = _get_cache_filename(url)
        # Hyphens, underscores, dots should be preserved
        assert "example-site.com" in result
        assert "_v1" in result

    def test_reversibility_principle(self):
        """Test that encoding preserves uniqueness (different URLs = different filenames)."""
        urls = [
            "https://example.com/api/v1",
            "https://example.com/api/v2",
            "https://example.com:8080/api/v1",
            "https://example.com/api/v1?param=1",
            "https://example.com/api/v1?param=2"
        ]
        filenames = [_get_cache_filename(url) for url in urls]
        # All filenames should be unique
        assert len(set(filenames)) == len(filenames)


if __name__ == '__main__':
    pytest.main([__file__])
