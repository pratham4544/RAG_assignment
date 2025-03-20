import os
import pickle
import hashlib

class Cache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, data: str) -> str:
        """Generate a unique cache key for the given data."""
        return hashlib.md5(data.encode()).hexdigest()
        
    def get(self, key: str) -> any:
        """Retrieve data from cache."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def set(self, key: str, data: any):
        """Store data in cache."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f) 