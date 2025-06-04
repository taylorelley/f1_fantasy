import os
import pandas as pd

class DataCache:
    """Simple cache for loading CSV files only when they change."""
    def __init__(self):
        self._cache = {}

    def load_csv(self, path, **kwargs):
        mtime = None
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            pass
        entry = self._cache.get(path)
        if entry and entry['mtime'] == mtime:
            return entry['df']
        df = pd.read_csv(path, **kwargs)
        self._cache[path] = {'mtime': mtime, 'df': df}
        return df

    def clear(self, base_path=None):
        if base_path:
            base_path = os.path.abspath(base_path)
            self._cache = {p: e for p, e in self._cache.items() if not os.path.abspath(p).startswith(base_path)}
        else:
            self._cache.clear()

