import pytest
from python_fbas.pubnet_data import _get_cache_filename


class TestFilenameGeneration:
    def test_basic_url_conversion(self):
        """The filename starts with a readable rendering of the URL."""
        url = "https://radar.withobsrvr.com/api/v1/node"
        result = _get_cache_filename(url)
        assert result.startswith(
            "stellar_network_data_radar.withobsrvr.com%2Fapi%2Fv1%2Fnode")
        assert result.endswith(".json")

    def test_deterministic(self):
        """The same URL always maps to the same filename."""
        url = "https://api.stellaratlas.io/v1/node"
        assert _get_cache_filename(url) == _get_cache_filename(url)

    def test_url_with_port(self):
        url = "https://example.com:8080/api/v1/node"
        assert "example.com%3A8080%2Fapi%2Fv1%2Fnode" in _get_cache_filename(url)

    def test_url_with_query_params(self):
        url = "https://example.com/api/v1/node?param1=value1&param2=value2"
        result = _get_cache_filename(url)
        assert "node%3Fparam1%3Dvalue1%26param2%3Dvalue2" in result

    def test_http_and_https_urls_differ(self):
        """http:// and https:// versions of a URL must not share a cache file."""
        assert _get_cache_filename("http://example.com/api/v1/node") != \
            _get_cache_filename("https://example.com/api/v1/node")

    def test_long_url_truncation(self):
        """Very long URLs produce filenames of bounded length."""
        url = ("https://very-long-domain-name-that-exceeds-normal-length.example.com/"
               "very/long/path/with/many/segments/that/should/be/truncated/to/"
               "reasonable/length/api/v1/node")
        result = _get_cache_filename(url)
        assert result.startswith("stellar_network_data_")
        assert result.endswith(".json")
        # Total length should be reasonable for all filesystems
        assert len(result) <= 250

    def test_long_urls_differing_past_truncation_point_differ(self):
        """URLs that only differ beyond the truncation point must not collide."""
        base = "https://example.com/" + "x" * 300
        assert _get_cache_filename(base + "/A") != _get_cache_filename(base + "/B")

    def test_special_characters(self):
        """Special characters in the URL get percent-encoded."""
        url = "https://example.com/path with spaces/file.json"
        result = _get_cache_filename(url)
        assert "example.com%2Fpath%20with%20spaces%2Ffile.json" in result

    def test_international_domain(self):
        """International domain names are properly encoded."""
        url = "https://xn--d1acpjx3f.xn--p1ai/api/nodes"  # пример.рф in punycode
        result = _get_cache_filename(url)
        assert result.startswith("stellar_network_data_")
        assert result.endswith(".json")
        # Should contain the punycode domain
        assert "xn--d1acpjx3f.xn--p1ai" in result

    def test_all_safe_characters_preserved(self):
        """Safe characters are not encoded unnecessarily."""
        url = "https://example-site.com/api_v1/nodes.json"
        result = _get_cache_filename(url)
        # Hyphens, underscores, dots should be preserved
        assert "example-site.com" in result
        assert "_v1" in result

    def test_uniqueness(self):
        """Different URLs map to different filenames."""
        urls = [
            "https://example.com/api/v1",
            "https://example.com/api/v2",
            "http://example.com/api/v1",
            "https://example.com:8080/api/v1",
            "https://example.com/api/v1?param=1",
            "https://example.com/api/v1?param=2"
        ]
        filenames = [_get_cache_filename(url) for url in urls]
        # All filenames should be unique
        assert len(set(filenames)) == len(filenames)


if __name__ == '__main__':
    pytest.main([__file__])
