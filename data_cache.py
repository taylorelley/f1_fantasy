# Simple in-memory cache for dataframes
import os
import pandas as pd
from typing import Dict, Optional

class DataCache:
    """Caches loaded DataFrames keyed by folder path."""
    _cache: Dict[str, Dict] = {}

    @classmethod
    def _normalize(cls, folder: str) -> str:
        if not folder.endswith('/'):
            folder += '/'
        return folder

    @classmethod
    def load_raw(cls, folder: str) -> Dict[str, pd.DataFrame]:
        """Load raw data CSVs from a folder, using cache if unchanged."""
        folder = cls._normalize(folder)
        files = [
            'driver_race_data.csv',
            'constructor_race_data.csv',
            'calendar.csv',
            'tracks.csv'
        ]
        mod_times = {f: os.path.getmtime(os.path.join(folder, f)) for f in files}
        entry = cls._cache.get(folder)
        if entry and entry.get('mod_times') == mod_times:
            return entry['raw']
        data = {f: pd.read_csv(os.path.join(folder, f)) for f in files}
        cls._cache[folder] = {'mod_times': mod_times, 'raw': data}
        return data

    @classmethod
    def store_processed(cls, folder: str, key: str, value):
        folder = cls._normalize(folder)
        cls._cache.setdefault(folder, {})[key] = value

    @classmethod
    def get_processed(cls, folder: str, key: str):
        folder = cls._normalize(folder)
        entry = cls._cache.get(folder)
        if entry:
            return entry.get(key)
        return None

    @classmethod
    def clear(cls, folder: Optional[str] = None):
        if folder:
            folder = cls._normalize(folder)
            cls._cache.pop(folder, None)
        else:
            cls._cache.clear()

