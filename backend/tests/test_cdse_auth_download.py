import json
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.product_downloader import get_product_downloader

client = TestClient(app)


def test_cdse_authentication_verification():
    """
    Verify CDSE OAuth client-credentials token acquisition without exposing credentials.
    Confirms token acquisition succeeds and returns safe metadata.
    """
    downloader = get_product_downloader()
    
    # 1. Credentials loaded from environment
    assert downloader.is_auth_configured() is True, "Copernicus credentials must be present in .env"
    
    # 2. Token acquisition succeeds
    auth_status = downloader.verify_authentication()
    assert auth_status["authenticated"] is True
    assert auth_status["token_acquired"] is True
    assert auth_status["flow"] in ["client_credentials", "password"]
    assert auth_status["token_length_chars"] > 500
    assert auth_status["expires_in_seconds"] is not None
    assert auth_status["expires_in_seconds"] > 0
    
    # 3. Security check: Ensure token string or secrets are not present in dictionary
    assert "token" not in auth_status
    assert "client_secret" not in auth_status
    assert "password" not in auth_status

    # 4. API endpoint verification
    response = client.get("/satellites/auth/verify")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["token_acquired"] is True
    assert data["token_length_chars"] > 500
    assert "token" not in data


def test_real_sentinel1_binary_download():
    """
    Perform a real download test of the Nepal 2026 pre-event Sentinel-1 scene:
    S1D_IW_GRDH_1SDV_20260819T001036_20260819T001101_004187_007ABD_DC16.SAFE

    Verifies:
    1. CDSE entity GUID is resolved from the live catalog.
    2. Actual binary bytes are downloaded over HTTPS with Bearer authentication.
    3. The file starts with the standard PK zip archive signature (b"PK\x03\x04").
    4. Data is stored in data/raw/sentinel1/ with SHA-256 and metadata sidecar.
    """
    downloader = get_product_downloader()
    product_name = "S1D_IW_GRDH_1SDV_20260819T001036_20260819T001101_004187_007ABD_DC16.SAFE"
    
    # Stream real binary satellite data (5 MB stream sample to verify binary pipe)
    download_size = 5 * 1024 * 1024  # 5 Megabytes
    result = downloader.download_product(
        product_id=product_name,
        collection="SENTINEL-1",
        max_bytes=download_size,
        force=True,
    )

    assert result.status == "completed", f"Download failed: {result.message}"
    assert result.file_path is not None
    assert result.file_size_bytes is not None
    assert result.file_size_bytes >= download_size
    assert result.checksum_sha256 is not None
    assert result.cdse_id == "f2682536-f9b8-4b5b-b9eb-5b52417820fa"

    downloaded_file = Path(result.file_path)
    assert downloaded_file.exists(), f"Target file does not exist on disk: {downloaded_file}"
    assert downloaded_file.stat().st_size >= download_size

    # Verify actual binary satellite data received (ZIP container magic bytes)
    with open(downloaded_file, "rb") as f:
        magic = f.read(4)
        assert magic.startswith(b"PK\x03\x04"), f"Expected ZIP magic bytes PK\\x03\\x04, got {magic!r}"

    # Verify metadata sidecar
    meta_file = downloaded_file.with_suffix(".meta.json")
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f_meta:
        meta_data = json.load(f_meta)
        assert meta_data["status"] == "completed"
        assert meta_data["is_valid_zip"] is True
        assert meta_data["cdse_id"] == "f2682536-f9b8-4b5b-b9eb-5b52417820fa"
