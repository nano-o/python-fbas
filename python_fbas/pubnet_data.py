"""
Module to fetch data from a source conforming to the stellarbeat.io API.

"""

import json
import os
import logging
import hashlib
from datetime import datetime
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
    """Generate a unique cache filename for a given URL."""
    # Create a hash of the URL to use as filename
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"stellar_network_data_{url_hash}.json"


def _migrate_old_cache(cache_dir: str, current_url: str) -> None:
    """Migrate old single cache file to new per-URL cache system."""
    old_path = os.path.join(cache_dir, 'stellar_network_data.json')
    if not os.path.exists(old_path):
        return
    
    try:
        with open(old_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # If it's the old format (just validators array), we can't migrate
        # since we don't know which URL it came from
        if isinstance(cache_data, list):
            print("Cache: Removing old cache format file...")
            os.remove(old_path)
            return
        
        # If it's new format but in old filename, migrate to new filename
        cached_url = cache_data.get('source_url')
        if cached_url:
            new_filename = _get_cache_filename(cached_url)
            new_path = os.path.join(cache_dir, new_filename)
            if not os.path.exists(new_path):
                print(f"Cache: Migrating cache data for {cached_url} to new format...")
                with open(new_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f)
            os.remove(old_path)
        else:
            # Unknown format, just remove
            os.remove(old_path)
    except Exception:
        # If migration fails, just remove the old file
        try:
            os.remove(old_path)
        except Exception:
            pass


def get_pubnet_config(update=False, url: str = None) -> list[dict]:
    """
    When update is true, fetch new data from the url and update the cache file for that URL.
    Otherwise, use cached data for the URL if available, or fetch fresh data if not cached.
    Each URL gets its own cache file that persists until explicitly updated.
    """
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    current_url = url if url is not None else get().stellar_data_url
    
    # Migrate old cache format if present
    _migrate_old_cache(cache_dir, current_url)
    
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
        print(f"Updating cache for URL: {current_url}")
        print(f"Cache file: {path}")
        _validators = _fetch_from_url(current_url)
        update_cache_file(_validators)
        return _validators
    
    # Check if cache exists for this URL
    try:
        logging.info("Reading Stellar network data from %s", path)
        with open(path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
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
