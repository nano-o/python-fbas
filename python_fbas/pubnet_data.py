"""
Module to fetch data from a source conforming to the stellarbeat.io API.

"""

import hashlib
import json
import os
import sys
import logging
from datetime import datetime
from urllib.parse import quote
from requests import get as http_get
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError
from platformdirs import user_cache_dir
from python_fbas.config import get

def _fetch_from_url(url: str | None = None) -> list[dict]:
    """
    Get data from url defined in the config file or provided url.
    """
    if url is None:
        url = get().stellar_data_url
    logging.info("Fetching data from %s", url)
    try:
        response = http_get(url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        raise IOError(
            f"Failed to fetch Stellar network data from {url}: {e}") from e
    if response.status_code != 200:
        raise IOError(
            f"Failed to fetch Stellar network data from {url}: "
            f"unexpected HTTP status {response.status_code}")
    try:
        data = response.json()
    except RequestsJSONDecodeError as e:
        raise ValueError(
            f"URL {url} did not return valid JSON data. Please check that the URL points to a Stellar network API endpoint") from e
    if not isinstance(data, list):
        raise ValueError(
            f"URL {url} returned invalid data format. Expected a JSON array of validators, got {type(data).__name__}")
    return data

def _get_cache_filename(url: str) -> str:
    """Generate a unique cache filename for a given URL.

    The filename is a human-readable, percent-encoded (RFC 3986) rendering of
    the URL followed by a digest of the full URL. The digest is what
    guarantees that distinct URLs never share a cache file: the readable part
    alone is ambiguous because it omits the protocol and because long URLs
    are truncated.
    """
    # Remove protocol for a cleaner readable part
    url_without_protocol = url.replace('https://', '').replace('http://', '')

    # Use percent-encoding (RFC 3986) for cross-platform safety
    # The 'safe' parameter allows common URL characters that are also safe in
    # filenames
    encoded = quote(url_without_protocol, safe='-._~')

    # Truncate long URLs; uniqueness is ensured by the digest
    max_length = 150  # conservative limit for most filesystems
    encoded = encoded[:max_length]

    digest = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
    return f"stellar_network_data_{encoded}_{digest}.json"


def get_pubnet_config(update=False, url: str | None = None) -> list[dict]:
    """
    When update is true, fetch new data from the url and update the cache file for that URL.
    Otherwise, use cached data for the URL if available, or fetch fresh data if not cached.
    Each URL gets its own cache file that persists until explicitly updated.
    """
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    current_url = url if url is not None else get().stellar_data_url
    
    # Get URL-specific cache filename
    cache_filename = _get_cache_filename(current_url)
    path = os.path.join(cache_dir, cache_filename)
    
    def update_cache_file(_validators):
        logging.info("Writing Stellar network data at %s", path)
        cache_data = {
            'source_url': current_url,
            'cached_at': datetime.now().isoformat(),
            'validators': _validators
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
    
    if update:
        try:
            print(f"Updating cache for URL: {current_url}")
            _validators = _fetch_from_url(current_url)
            update_cache_file(_validators)
            print(f"Cache file: {path}")
            return _validators
        except Exception as e:
            print(f"Failed to update cache for URL: {current_url}", file=sys.stderr)
            raise
    
    # Check if cache exists for this URL
    try:
        logging.info("Reading Stellar network data from %s", path)
        with open(path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        if (not isinstance(cache_data, dict)
                or 'validators' not in cache_data
                or not isinstance(cache_data['validators'], list)):
            raise ValueError("Cache data missing 'validators' list")

        # Never serve data that was fetched from a different URL:
        if cache_data.get('source_url') != current_url:
            raise ValueError(
                f"Cache file was created for URL "
                f"{cache_data.get('source_url')!r}, not {current_url!r}")

        # Use cached data
        cached_at = cache_data.get('cached_at', 'unknown time')
        if cached_at != 'unknown time':
            try:
                cache_time = datetime.fromisoformat(cached_at)
                print(f"Cache: Using cached data for {current_url} from {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
            except ValueError:
                print(f"Cache: Using cached data for {current_url} from unknown time")
        else:
            print(f"Cache: Using cached data for {current_url}")
        return cache_data['validators']
        
    except FileNotFoundError:
        print(f"Cache: No cache found for {current_url}, fetching fresh data...")
        _validators = _fetch_from_url(current_url)
        update_cache_file(_validators)
        print(f"Cache: Saved fresh data to cache for {current_url}")
        return _validators
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logging.warning("Ignoring invalid cache file %s: %s", path, e)
        print(f"Cache: Invalid cache for {current_url}, fetching fresh data...")
        _validators = _fetch_from_url(current_url)
        update_cache_file(_validators)
        print(f"Cache: Saved fresh data to cache for {current_url}")
        return _validators
