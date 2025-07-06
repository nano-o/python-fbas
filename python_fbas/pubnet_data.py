"""
Module to fetch data from a source conforming to the stellarbeat.io API.

"""

import json
import os
import logging
from requests import get as http_get
from platformdirs import user_cache_dir
from python_fbas.config import get

def _fetch_from_url() -> list[dict]:
    """
    Get data from url defined in the config file.
    """
    url = get().stellar_data_url
    logging.info("Fetching data from %s", url)
    response = http_get(url, timeout=5)
    if response.status_code == 200:
        return response.json()
    response.raise_for_status()
    raise IOError("Failed to fetch Stellar network data")

def get_pubnet_config(update=False) -> list[dict]:
    """
    When update is true, fetch new data from the url and update the file in the cache directory.
    """
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    path = os.path.join(cache_dir, 'stellar_network_data.json')
    def update_cache_file(_validators):
        logging.info("Writing Stellar network data at %s", path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_validators, f)
    if update:
        print(f"Updating cache at {path}")
        _validators = _fetch_from_url()
        update_cache_file(_validators)
    else:
        try:
            logging.info("Reading Stellar network data from %s", path)
            with open(path, 'r', encoding='utf-8') as f:
                _validators = json.load(f)
        except FileNotFoundError:
            logging.info("Cache file not found")
            _validators = _fetch_from_url()
            update_cache_file(_validators)
    return _validators
