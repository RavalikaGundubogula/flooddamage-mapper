import re
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class GeoJSONPolygon(BaseModel):
    """GeoJSON-compliant Polygon geometry representation."""
    type: Literal["Polygon"] = "Polygon"
    coordinates: List[List[List[float]]]


def validate_coordinates_range(lon: float, lat: float) -> None:
    """Validate longitude [-180, 180] and latitude [-90, 90]."""
    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
        raise ValueError(f"Coordinates must be numeric, got lon={lon}, lat={lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude must be between -180.0 and 180.0 degrees, got {lon}")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be between -90.0 and 90.0 degrees, got {lat}")


def bbox_to_polygon_coordinates(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> List[List[List[float]]]:
    """Convert bounding box coordinates to a closed GeoJSON Polygon exterior ring."""
    validate_coordinates_range(min_lon, min_lat)
    validate_coordinates_range(max_lon, max_lat)

    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})")

    # Counter-clockwise exterior ring: SW -> SE -> NE -> NW -> SW
    ring = [
        [float(min_lon), float(min_lat)],
        [float(max_lon), float(min_lat)],
        [float(max_lon), float(max_lat)],
        [float(min_lon), float(max_lat)],
        [float(min_lon), float(min_lat)],
    ]
    return [ring]


def normalize_and_validate_aoi(aoi: Any) -> Dict[str, Any]:
    """
    Normalize any supported AOI format into a GeoJSON-compatible Polygon dict.
    Supports:
    - GeoJSON Polygon: {"type": "Polygon", "coordinates": [...]}
    - GeoJSON Feature: {"type": "Feature", "geometry": {"type": "Polygon", ...}}
    - Bounding Box dict: {"bbox": [min_lon, min_lat, max_lon, max_lat]}
    - Named Bounding Box dict: {"min_lon": ..., "min_lat": ..., "max_lon": ..., "max_lat": ...}
    - Bounding Box list: [min_lon, min_lat, max_lon, max_lat]
    """
    if not aoi:
        raise ValueError("AOI cannot be empty")

    # 1. Bounding box as a flat list: [min_lon, min_lat, max_lon, max_lat]
    if isinstance(aoi, (list, tuple)):
        if len(aoi) == 4 and all(isinstance(x, (int, float)) for x in aoi):
            coords = bbox_to_polygon_coordinates(aoi[0], aoi[1], aoi[2], aoi[3])
            return {"type": "Polygon", "coordinates": coords}
        raise ValueError("List-based AOI must be a 4-element bounding box [min_lon, min_lat, max_lon, max_lat]")

    if not isinstance(aoi, dict):
        raise ValueError("AOI must be a valid GeoJSON Polygon object or Bounding Box")

    # 2. GeoJSON Feature
    if aoi.get("type") == "Feature":
        geom = aoi.get("geometry")
        if not isinstance(geom, dict):
            raise ValueError("Feature AOI must contain a 'geometry' object")
        aoi = geom

    # 3. Explicit bbox dict: {"bbox": [min_lon, min_lat, max_lon, max_lat]} or {"type": "bbox", ...}
    if "bbox" in aoi:
        bbox_val = aoi["bbox"]
        if not isinstance(bbox_val, (list, tuple)) or len(bbox_val) != 4:
            raise ValueError("Bounding box 'bbox' must be a list of 4 numbers [min_lon, min_lat, max_lon, max_lat]")
        coords = bbox_to_polygon_coordinates(bbox_val[0], bbox_val[1], bbox_val[2], bbox_val[3])
        return {"type": "Polygon", "coordinates": coords}

    # 4. Named coordinate dict: {"min_lon": ..., "min_lat": ..., "max_lon": ..., "max_lat": ...}
    named_keys = {"min_lon", "min_lat", "max_lon", "max_lat"}
    if named_keys.issubset(aoi.keys()):
        coords = bbox_to_polygon_coordinates(
            aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"]
        )
        return {"type": "Polygon", "coordinates": coords}

    # Also handle south, west, north, east
    alt_keys = {"west", "south", "east", "north"}
    if alt_keys.issubset(aoi.keys()):
        coords = bbox_to_polygon_coordinates(
            aoi["west"], aoi["south"], aoi["east"], aoi["north"]
        )
        return {"type": "Polygon", "coordinates": coords}

    # 5. GeoJSON Polygon: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
    if aoi.get("type") == "Polygon" or "coordinates" in aoi:
        coordinates = aoi.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) == 0:
            raise ValueError("Polygon must contain a non-empty 'coordinates' list")

        normalized_rings = []
        for ring_idx, ring in enumerate(coordinates):
            if not isinstance(ring, list) or len(ring) < 4:
                raise ValueError(
                    f"Polygon ring {ring_idx} must contain at least 4 coordinates (3 positions + closure)"
                )

            normalized_ring = []
            for pt_idx, pt in enumerate(ring):
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    raise ValueError(
                        f"Point {pt_idx} in ring {ring_idx} must be a [longitude, latitude] pair"
                    )
                lon, lat = float(pt[0]), float(pt[1])
                validate_coordinates_range(lon, lat)
                normalized_ring.append([lon, lat])

            # Validate linear ring closure (first point equals last point)
            if normalized_ring[0] != normalized_ring[-1]:
                raise ValueError(
                    f"Polygon linear ring {ring_idx} must be closed (first and last coordinates must match)"
                )

            normalized_rings.append(normalized_ring)

        return {"type": "Polygon", "coordinates": normalized_rings}

    raise ValueError(
        "Invalid AOI structure. Must be a GeoJSON Polygon or a Bounding Box ([min_lon, min_lat, max_lon, max_lat])"
    )


class AnalysisCreateRequest(BaseModel):
    """User input request for starting a new flood damage analysis."""
    aoi: Any = Field(
        ...,
        description="Area of Interest: either a GeoJSON Polygon or a Bounding Box"
    )
    flood_date: str = Field(
        ...,
        description="Event/Flood date in YYYY-MM-DD format",
        examples=["2024-05-15"]
    )
    report_language: str = Field(
        default="en",
        description="Language code for the analysis report (default: 'en')",
        examples=["en"]
    )

    @field_validator("flood_date")
    @classmethod
    def validate_flood_date(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("flood_date must be a string in YYYY-MM-DD format")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v.strip()):
            raise ValueError("flood_date must follow YYYY-MM-DD format (e.g. 2024-05-15)")
        try:
            parsed = date.fromisoformat(v.strip())
        except ValueError as e:
            raise ValueError(f"Invalid calendar date: {v}. Details: {e}")

        if parsed.year < 1970 or parsed.year > 2100:
            raise ValueError(f"flood_date year must be between 1970 and 2100, got {parsed.year}")

        return v.strip()

    @field_validator("report_language")
    @classmethod
    def validate_report_language(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            return "en"
        if not re.match(r"^[a-z]{2,5}(-[a-z]{2,5})?$", cleaned):
            raise ValueError(f"Invalid report_language code: '{v}'. Use 2-letter ISO code like 'en', 'es', 'fr'")
        return cleaned

    @model_validator(mode="after")
    def validate_and_normalize_aoi(self) -> "AnalysisCreateRequest":
        # Normalize and validate AOI, transforming bbox into Polygon if needed
        self.aoi = normalize_and_validate_aoi(self.aoi)
        return self


class AnalysisResponse(BaseModel):
    """Response returned upon creating or querying an analysis job."""
    job_id: str = Field(..., description="Unique analysis job identifier")
    status: str = Field(..., description="Current job status: created, processing, completed, failed")
    aoi: Dict[str, Any] = Field(..., description="Normalized GeoJSON Polygon AOI")
    flood_date: str = Field(..., description="Event date in YYYY-MM-DD format")
    report_language: str = Field(default="en", description="Target report language")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp (UTC)")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp (UTC)")
