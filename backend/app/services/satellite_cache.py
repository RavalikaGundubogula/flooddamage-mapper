import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from app.schemas.satellite import SceneMetadata

logger = logging.getLogger(__name__)


class SatelliteDiscoveryCache:
    """
    File-based caching layer for satellite catalog discovery queries.
    Stores normalized scene metadata in backend/data/cache to prevent
    redundant network round-trips to satellite catalogs.
    """

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: int = 86400) -> None:
        settings = get_settings()
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = settings.get_data_path(settings.DATA_CACHE_DIR)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _generate_cache_key(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        collection: str,
        max_results: int,
    ) -> str:
        """Create a deterministic hash key from query parameters."""
        normalized_query = {
            "collection": collection.upper().strip(),
            "start_date": start_date.strip(),
            "end_date": end_date.strip(),
            "max_results": max_results,
            "aoi": aoi,
        }
        serialized = json.dumps(normalized_query, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_cache_path(self, cache_key: str, collection: str) -> Path:
        """Get file path for a specific cache key."""
        filename = f"catalog_{collection.lower()}_{cache_key[:16]}.json"
        return self.cache_dir / filename

    def get(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        collection: str,
        max_results: int,
    ) -> Optional[List[SceneMetadata]]:
        """
        Retrieve cached scene metadata if available and not expired.
        Returns None on cache miss or expired TTL.
        """
        cache_key = self._generate_cache_key(aoi, start_date, end_date, collection, max_results)
        cache_path = self._get_cache_path(cache_key, collection)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check timestamp and TTL
            created_at_str = data.get("created_at")
            if not created_at_str:
                return None

            created_at = datetime.fromisoformat(created_at_str)
            age = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age > self.ttl_seconds:
                logger.debug("Cache entry %s expired (age: %ss, ttl: %ss)", cache_path.name, age, self.ttl_seconds)
                cache_path.unlink(missing_ok=True)
                return None

            raw_scenes = data.get("scenes", [])
            scenes = [SceneMetadata(**s) for s in raw_scenes]
            logger.debug("Cache HIT for %s (%d scenes loaded from %s)", collection, len(scenes), cache_path.name)
            return scenes

        except Exception as e:
            logger.warning("Failed to read or parse cache file %s: %s", cache_path, e)
            cache_path.unlink(missing_ok=True)
            return None

    def set(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        collection: str,
        max_results: int,
        scenes: List[SceneMetadata],
    ) -> None:
        """Persist discovery results to disk with metadata and timestamp."""
        cache_key = self._generate_cache_key(aoi, start_date, end_date, collection, max_results)
        cache_path = self._get_cache_path(cache_key, collection)

        payload = {
            "cache_key": cache_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": collection.upper(),
            "ttl_seconds": self.ttl_seconds,
            "query": {
                "start_date": start_date,
                "end_date": end_date,
                "max_results": max_results,
                "aoi": aoi,
            },
            "scene_count": len(scenes),
            "scenes": [s.model_dump(mode="json") for s in scenes],
        }

        try:
            # Write atomically to temporary file then replace
            temp_path = cache_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_path.replace(cache_path)
            logger.debug("Cached %d %s scenes to %s", len(scenes), collection, cache_path.name)
        except Exception as e:
            logger.warning("Failed to write discovery cache to %s: %s", cache_path, e)

    def clear(self) -> int:
        """Clear all catalog discovery cache files. Returns count of deleted files."""
        count = 0
        for f in self.cache_dir.glob("catalog_*.json"):
            try:
                f.unlink()
                count += 1
            except Exception as e:
                logger.warning("Error deleting cache file %s: %s", f, e)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Return cache directory statistics."""
        files = list(self.cache_dir.glob("catalog_*.json"))
        total_size = sum(f.stat().st_size for f in files) if files else 0
        return {
            "cached_queries_count": len(files),
            "total_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl_seconds,
        }


# Global singleton instance
_cache_instance = SatelliteDiscoveryCache()


def get_satellite_cache() -> SatelliteDiscoveryCache:
    """Dependency provider for cache."""
    return _cache_instance
