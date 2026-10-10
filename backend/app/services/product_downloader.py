import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.clients.copernicus import CopernicusCatalogClient, SatelliteCatalog
from app.core.config import Settings, get_settings
from app.schemas.download import (
    DownloadSummaryResponse,
    ProductDownloadRequest,
    ProductDownloadResult,
    StoredDownloadItem,
    StoredDownloadsResponse,
)
from app.schemas.scene_selection import SelectedScenePair

logger = logging.getLogger(__name__)


class ProductDownloaderService:
    """
    Service responsible for retrieving raw satellite products from the
    Copernicus Data Space Ecosystem (CDSE) Zipper API, managing local raw disk
    caches, verifying cryptographic checksums, and handling authentication.
    """

    def __init__(
        self,
        catalog_client: Optional[SatelliteCatalog] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog_client = catalog_client or CopernicusCatalogClient()
        self._cached_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    def is_auth_configured(self) -> bool:
        """Check whether credentials are provided via environment or settings."""
        has_user_pass = bool(self.settings.COPERNICUS_USERNAME and self.settings.COPERNICUS_PASSWORD)
        has_client_cred = bool(self.settings.COPERNICUS_CLIENT_ID and self.settings.COPERNICUS_CLIENT_SECRET)
        return has_user_pass or has_client_cred

    def obtain_access_token(self) -> Optional[str]:
        """
        Request OAuth2 Bearer token from CDSE Keycloak endpoint.
        Returns cached active token if still valid.
        """
        now = datetime.now(timezone.utc)
        if self._cached_token and self._token_expires_at and now < self._token_expires_at:
            return self._cached_token

        if not self.is_auth_configured():
            logger.info("Copernicus CDSE credentials not configured.")
            return None

        # Prepare request payload
        if self.settings.COPERNICUS_USERNAME and self.settings.COPERNICUS_PASSWORD:
            data = {
                "client_id": "cdse-public",
                "username": self.settings.COPERNICUS_USERNAME,
                "password": self.settings.COPERNICUS_PASSWORD,
                "grant_type": "password",
            }
        else:
            data = {
                "client_id": self.settings.COPERNICUS_CLIENT_ID,
                "client_secret": self.settings.COPERNICUS_CLIENT_SECRET,
                "grant_type": "client_credentials",
            }

        encoded_data = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(
            self.settings.COPERNICUS_AUTH_URL,
            data=encoded_data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "FloodDamageMapper/0.1",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    token = payload.get("access_token")
                    expires_in = int(payload.get("expires_in", 300))
                    self._cached_token = token
                    self._token_expires_at = now + timedelta(seconds=max(expires_in - 30, 30))
                    logger.info("Successfully acquired CDSE access token (expires in %ds)", expires_in)
                    return token
                logger.error("CDSE Auth returned HTTP %d", resp.status)
                return None
        except urllib.error.HTTPError as e:
            logger.error("CDSE Keycloak auth error: %d %s", e.code, e.reason)
            return None
        except Exception as e:
            logger.error("CDSE Keycloak connection failed: %s", e)
            return None

    def verify_authentication(self) -> Dict[str, Any]:
        """
        Verify CDSE OAuth authentication and token acquisition without exposing secrets.
        Returns safe status dictionary reporting success/failure and non-secret metadata.
        """
        if not self.is_auth_configured():
            return {
                "authenticated": False,
                "configured": False,
                "flow": None,
                "token_acquired": False,
                "message": "Copernicus CDSE credentials are not configured in environment.",
                "auth_endpoint": self.settings.COPERNICUS_AUTH_URL,
            }

        flow = "password" if (self.settings.COPERNICUS_USERNAME and self.settings.COPERNICUS_PASSWORD) else "client_credentials"
        token = self.obtain_access_token()
        if token:
            expires_in_sec = (
                int((self._token_expires_at - datetime.now(timezone.utc)).total_seconds())
                if self._token_expires_at
                else None
            )
            return {
                "authenticated": True,
                "configured": True,
                "flow": flow,
                "token_acquired": True,
                "token_length_chars": len(token),
                "expires_in_seconds": expires_in_sec,
                "auth_endpoint": self.settings.COPERNICUS_AUTH_URL,
            }

        return {
            "authenticated": False,
            "configured": True,
            "flow": flow,
            "token_acquired": False,
            "message": "Failed to acquire OAuth2 token from CDSE Keycloak.",
            "auth_endpoint": self.settings.COPERNICUS_AUTH_URL,
        }

    def _resolve_storage_directory(self, collection: str) -> Path:
        """Resolve storage directory for a collection, creating it if necessary."""
        c = collection.upper()
        if "SENTINEL-1" in c or c.startswith("S1"):
            target_dir = self.settings.get_data_path(f"{self.settings.DATA_RAW_DIR}/sentinel1")
        elif "SENTINEL-2" in c or c.startswith("S2"):
            target_dir = self.settings.get_data_path(f"{self.settings.DATA_RAW_DIR}/sentinel2")
        else:
            target_dir = self.settings.get_data_path(self.settings.DATA_RAW_DIR)
        
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _clean_product_name(self, product_id: str) -> str:
        """Clean and normalize product identifier for filesystem naming."""
        clean = product_id.strip()
        if clean.endswith(".SAFE"):
            clean = clean[:-5]
        return clean

    def check_existing_download(self, product_id: str, collection: str) -> Optional[ProductDownloadResult]:
        """Check if product is already cached locally on disk."""
        target_dir = self._resolve_storage_directory(collection)
        clean_name = self._clean_product_name(product_id)

        # Check for .zip, .SAFE, or directory
        candidates = [
            target_dir / f"{clean_name}.zip",
            target_dir / f"{clean_name}.SAFE.zip",
            target_dir / f"{clean_name}.SAFE",
            target_dir / clean_name,
        ]

        existing_path: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                existing_path = cand
                break

        if not existing_path:
            return None

        # Check metadata sidecar
        meta_file = target_dir / f"{clean_name}.meta.json"
        meta_data: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                pass

        file_size = (
            sum(f.stat().st_size for f in existing_path.rglob("*") if f.is_file())
            if existing_path.is_dir()
            else existing_path.stat().st_size
        )

        return ProductDownloadResult(
            product_id=product_id,
            cdse_id=meta_data.get("cdse_id"),
            collection=collection.upper(),
            status="cached",
            file_path=str(existing_path.resolve()),
            file_size_bytes=file_size,
            checksum_sha256=meta_data.get("checksum_sha256"),
            download_url=meta_data.get("download_url"),
            auth_configured=self.is_auth_configured(),
            message=f"Product is already cached locally at {existing_path.name}",
        )

    def download_product(
        self,
        product_id: str,
        collection: Optional[str] = None,
        max_bytes: Optional[int] = None,
        force: bool = False,
    ) -> ProductDownloadResult:
        """
        Download a single satellite product from Copernicus CDSE.
        Handles cache check, catalog UUID resolution, authentication enforcement,
        streaming download, and SHA-256 verification.
        """
        # Infer collection from product name if not provided
        if not collection:
            p_upper = product_id.upper()
            if p_upper.startswith("S1"):
                collection = "SENTINEL-1"
            elif p_upper.startswith("S2"):
                collection = "SENTINEL-2"
            else:
                collection = "SENTINEL-1"

        # 1. Check local disk cache first (unless forced)
        if not force:
            cached_result = self.check_existing_download(product_id, collection)
            if cached_result:
                return cached_result

        # 2. Query catalog to resolve CDSE UUID and metadata
        scene_meta = self.catalog_client.get_product(product_id)
        cdse_id = None
        content_length = None
        if scene_meta:
            collection = scene_meta.collection
            extra = scene_meta.extra_attributes or {}
            cdse_id = extra.get("cdse_id")
            content_length = extra.get("content_length")

        # If product was passed directly as a CDSE UUID
        is_uuid = bool(re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", product_id.strip()))
        if is_uuid and not cdse_id:
            cdse_id = product_id.strip()

        download_url = (
            f"{self.settings.COPERNICUS_DOWNLOAD_URL}({cdse_id})/$value"
            if cdse_id
            else None
        )

        # 3. Check authentication requirement
        if not self.is_auth_configured():
            logger.info("Copernicus credentials missing for download of %s", product_id)
            return ProductDownloadResult(
                product_id=product_id,
                cdse_id=cdse_id,
                collection=collection,
                status="authentication_required",
                file_path=None,
                file_size_bytes=content_length,
                checksum_sha256=None,
                download_url=download_url,
                auth_configured=False,
                message=(
                    "Copernicus Data Space Ecosystem (CDSE) credentials are required to download "
                    "raw satellite products. Register at https://dataspace.copernicus.eu/ and set "
                    "COPERNICUS_USERNAME and COPERNICUS_PASSWORD in backend/.env."
                ),
            )

        # 4. If credentials ARE configured, acquire token and stream download
        token = self.obtain_access_token()
        if not token:
            return ProductDownloadResult(
                product_id=product_id,
                cdse_id=cdse_id,
                collection=collection,
                status="failed",
                file_path=None,
                file_size_bytes=content_length,
                checksum_sha256=None,
                download_url=download_url,
                auth_configured=True,
                message="Failed to obtain OAuth2 token from CDSE Keycloak.",
            )

        if not cdse_id or not download_url:
            return ProductDownloadResult(
                product_id=product_id,
                cdse_id=None,
                collection=collection,
                status="failed",
                file_path=None,
                file_size_bytes=None,
                checksum_sha256=None,
                download_url=None,
                auth_configured=True,
                message=f"Could not resolve CDSE entity GUID for product: {product_id}",
            )

        # 5. Stream download to target raw path
        target_dir = self._resolve_storage_directory(collection)
        clean_name = self._clean_product_name(product_id)
        archive_file = target_dir / f"{clean_name}.zip"
        part_file = target_dir / f"{clean_name}.zip.part"
        meta_file = target_dir / f"{clean_name}.meta.json"

        req = urllib.request.Request(
            download_url,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "FloodDamageMapper/0.1",
            },
        )

        sha256_hash = hashlib.sha256()
        total_downloaded = 0
        chunk_size = 262144  # 256KB chunks for high network throughput

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                with open(part_file, "wb") as f_out:
                    while True:
                        if max_bytes is not None and total_downloaded >= max_bytes:
                            break
                        read_size = chunk_size
                        if max_bytes is not None:
                            read_size = min(chunk_size, max_bytes - total_downloaded)
                        chunk = resp.read(read_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        sha256_hash.update(chunk)
                        total_downloaded += len(chunk)

            # Rename part to final target
            if part_file.exists():
                part_file.replace(archive_file)

            computed_sha256 = sha256_hash.hexdigest()

            # Verify binary ZIP format header (PK magic bytes)
            is_valid_zip = False
            if archive_file.exists() and archive_file.stat().st_size >= 4:
                with open(archive_file, "rb") as f_chk:
                    is_valid_zip = f_chk.read(4).startswith(b"PK")

            # Write sidecar metadata
            meta_payload = {
                "product_id": product_id,
                "clean_name": clean_name,
                "cdse_id": cdse_id,
                "collection": collection,
                "download_url": download_url,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "file_size_bytes": total_downloaded,
                "checksum_sha256": computed_sha256,
                "is_valid_zip": is_valid_zip,
                "status": "completed",
            }
            with open(meta_file, "w", encoding="utf-8") as f_meta:
                json.dump(meta_payload, f_meta, indent=2)

            return ProductDownloadResult(
                product_id=product_id,
                cdse_id=cdse_id,
                collection=collection,
                status="completed",
                file_path=str(archive_file.resolve()),
                file_size_bytes=total_downloaded,
                checksum_sha256=computed_sha256,
                download_url=download_url,
                auth_configured=True,
                message=f"Product downloaded successfully ({total_downloaded} bytes, valid_zip={is_valid_zip}).",
            )
        except Exception as e:
            if part_file.exists():
                try:
                    part_file.unlink()
                except Exception:
                    pass
            logger.error("Download failed for product %s: %s", product_id, e)
            return ProductDownloadResult(
                product_id=product_id,
                cdse_id=cdse_id,
                collection=collection,
                status="failed",
                file_path=None,
                file_size_bytes=None,
                checksum_sha256=None,
                download_url=download_url,
                auth_configured=True,
                message=f"Product download error: {str(e)}",
            )

    def download_pair(self, pair: SelectedScenePair) -> List[ProductDownloadResult]:
        """Download both before and after scenes of a selected pair."""
        results: List[ProductDownloadResult] = []
        # Before scene
        results.append(
            self.download_product(
                product_id=pair.before_scene.product_id,
                collection=pair.satellite,
            )
        )
        # After scene
        results.append(
            self.download_product(
                product_id=pair.after_scene.product_id,
                collection=pair.satellite,
            )
        )
        return results

    def execute_download_request(
        self,
        request: ProductDownloadRequest,
    ) -> DownloadSummaryResponse:
        """Process a high-level download request containing products, pair, or single ID."""
        results: List[ProductDownloadResult] = []

        if request.scene_pair:
            results.extend(self.download_pair(request.scene_pair))

        if request.product_ids:
            for pid in request.product_ids:
                results.append(self.download_product(pid, request.collection))

        if request.product_id and not request.product_ids and not request.scene_pair:
            results.append(self.download_product(request.product_id, request.collection))

        completed = sum(1 for r in results if r.status == "completed")
        cached = sum(1 for r in results if r.status == "cached")
        auth_req = sum(1 for r in results if r.status == "authentication_required")
        failed = sum(1 for r in results if r.status == "failed")

        return DownloadSummaryResponse(
            total_requested=len(results),
            completed_count=completed,
            cached_count=cached,
            auth_required_count=auth_req,
            failed_count=failed,
            auth_configured=self.is_auth_configured(),
            results=results,
        )

    def list_downloaded_products(self) -> StoredDownloadsResponse:
        """Scan raw storage directories and return catalog of local satellite products."""
        s1_dir = self.settings.get_data_path(f"{self.settings.DATA_RAW_DIR}/sentinel1")
        s2_dir = self.settings.get_data_path(f"{self.settings.DATA_RAW_DIR}/sentinel2")

        items: List[StoredDownloadItem] = []
        s1_count = 0
        s2_count = 0
        total_size = 0

        for target_dir, coll in [(s1_dir, "SENTINEL-1"), (s2_dir, "SENTINEL-2")]:
            if not target_dir.exists():
                continue
            for entry in target_dir.iterdir():
                if entry.name.endswith(".meta.json") or entry.name == ".gitkeep":
                    continue
                
                size = (
                    sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                    if entry.is_dir()
                    else entry.stat().st_size
                )
                total_size += size

                if coll == "SENTINEL-1":
                    s1_count += 1
                else:
                    s2_count += 1

                # Check for companion metadata
                clean_name = self._clean_product_name(entry.name)
                meta_file = target_dir / f"{clean_name}.meta.json"
                sha256 = None
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            sha256 = m.get("checksum_sha256")
                    except Exception:
                        pass

                ctime = datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc)

                items.append(
                    StoredDownloadItem(
                        product_id=entry.name,
                        collection=coll,
                        file_name=entry.name,
                        file_path=str(entry.resolve()),
                        file_size_bytes=size,
                        created_at=ctime,
                        checksum_sha256=sha256,
                        metadata_path=str(meta_file.resolve()) if meta_file.exists() else None,
                    )
                )

        return StoredDownloadsResponse(
            total_products=len(items),
            total_size_bytes=total_size,
            sentinel1_count=s1_count,
            sentinel2_count=s2_count,
            products=items,
        )


_downloader_instance: Optional[ProductDownloaderService] = None


def get_product_downloader() -> ProductDownloaderService:
    """Return singleton instance of ProductDownloaderService."""
    global _downloader_instance
    if _downloader_instance is None:
        _downloader_instance = ProductDownloaderService()
    return _downloader_instance
