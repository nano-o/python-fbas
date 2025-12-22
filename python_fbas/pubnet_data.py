"""
Module to fetch data from a source conforming to the stellarbeat.io API.

"""

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

def _fetch_from_url(url: str = None) -> list[dict]:
    """
    Get data from url defined in the config file or provided url.
    """
    if url is None:
        url = get().stellar_data_url
    logging.info("Fetching data from %s", url)
    try:
        response = http_get(url, timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                if not isinstance(data, list):
                    raise ValueError(f"URL {url} returned invalid data format. Expected a JSON array of validators, got {type(data).__name__}")
                return data
            except (RequestsJSONDecodeError, ValueError) as e:
                if isinstance(e, RequestsJSONDecodeError):
                    raise ValueError(f"URL {url} did not return valid JSON data. Please check that the URL points to a Stellar network API endpoint") from e
                raise
        response.raise_for_status()
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise IOError(f"Failed to fetch Stellar network data from {url}: {e}") from e

def _get_cache_filename(url: str) -> str:
    """Generate a cache filename for a given URL using RFC 3986 percent-encoding.
    
    This approach is cross-platform and standards-compliant, ensuring that:
    - All special characters are properly encoded
    - Filenames work on Windows, macOS, and Linux
    - The encoding is reversible and unambiguous
    - Very long URLs are handled gracefully
    """
    # Remove protocol for cleaner filenames
    url_without_protocol = url.replace('https://', '').replace('http://', '')
    
    # Use percent-encoding (RFC 3986) for cross-platform safety
    # The 'safe' parameter allows common URL characters that are also safe in filenames
    encoded = quote(url_without_protocol, safe='-._~')
    
    # Handle very long URLs by truncating if necessary
    max_length = 200  # Conservative limit for most filesystems
    if len(encoded) > max_length:
        # Keep the domain part if possible
        if '/' in url_without_protocol:
            domain, path = url_without_protocol.split('/', 1)
            domain_encoded = quote(domain, safe='-._~')
            remaining_length = max_length - len(domain_encoded) - 1  # -1 for separator
            if remaining_length > 10:  # Ensure we have meaningful path info
                path_encoded = quote(path, safe='-._~')[:remaining_length]
                encoded = f"{domain_encoded}_{path_encoded}"
            else:
                encoded = domain_encoded
        else:
            # Just truncate if no path separator
            encoded = encoded[:max_length]
    
    return f"stellar_network_data_{encoded}.json"


def get_pubnet_config(update=False, url: str = None) -> list[dict]:
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
