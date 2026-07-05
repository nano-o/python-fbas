"""Unit tests for pubnet_data. No network access: HTTP and the cache
directory are mocked/redirected."""

import json
import pytest

import python_fbas.pubnet_data as pubnet_data


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_fetch_non_200_success_status_raises(monkeypatch):
    """A 2xx status other than 200 (e.g. 204) must raise, not return None."""
    monkeypatch.setattr(pubnet_data, 'http_get',
                        lambda url, timeout: FakeResponse(status_code=204))
    with pytest.raises(IOError, match="unexpected HTTP status 204"):
        pubnet_data._fetch_from_url("https://example.com/api")


def test_fetch_non_list_json_raises_value_error(monkeypatch):
    """Valid JSON that is not a list of validators must raise ValueError."""
    monkeypatch.setattr(
        pubnet_data, 'http_get',
        lambda url, timeout: FakeResponse(json_data={"not": "a list"}))
    with pytest.raises(ValueError, match="invalid data format"):
        pubnet_data._fetch_from_url("https://example.com/api")


def test_cache_for_same_url_is_used(monkeypatch, tmp_path):
    url = "https://example.com/a"
    monkeypatch.setattr(pubnet_data, 'user_cache_dir',
                        lambda *a, **k: str(tmp_path))
    cached = [{'publicKey': 'CACHED'}]
    path = tmp_path / pubnet_data._get_cache_filename(url)
    path.write_text(json.dumps({
        'source_url': url,
        'cached_at': '2026-01-01T00:00:00',
        'validators': cached,
    }))

    def fail_fetch(url=None):
        raise AssertionError("must not fetch when a valid cache exists")
    monkeypatch.setattr(pubnet_data, '_fetch_from_url', fail_fetch)

    assert pubnet_data.get_pubnet_config(url=url) == cached


def test_cache_for_different_url_is_refetched(monkeypatch, tmp_path):
    """A cache file whose source_url does not match must be ignored."""
    url_a = "https://example.com/a"
    url_b = "https://example.com/b"
    monkeypatch.setattr(pubnet_data, 'user_cache_dir',
                        lambda *a, **k: str(tmp_path))
    # Simulate a stale or colliding cache file: it sits at url_a's path but
    # holds data fetched from url_b.
    path = tmp_path / pubnet_data._get_cache_filename(url_a)
    path.write_text(json.dumps({
        'source_url': url_b,
        'cached_at': '2026-01-01T00:00:00',
        'validators': [{'publicKey': 'STALE'}],
    }))
    fresh = [{'publicKey': 'FRESH'}]
    monkeypatch.setattr(pubnet_data, '_fetch_from_url', lambda url=None: fresh)

    assert pubnet_data.get_pubnet_config(url=url_a) == fresh
    # The cache file must have been rewritten for the right URL:
    rewritten = json.loads(path.read_text())
    assert rewritten['source_url'] == url_a
    assert rewritten['validators'] == fresh


if __name__ == '__main__':
    pytest.main([__file__])
