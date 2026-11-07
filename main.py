from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).resolve().parent / ".env")

from ai_service import (
    AiServiceError,
    extract_address_point,
    extract_address_points_from_images,
)
from geocoding_service import GeocodingError, geocode
from routing_service import (
    RoutingError,
    calculate_route,
    optimize_route,
)
from schemas import (
    VEHICLE_PRESETS,
    ExtractedAddress,
    ExtractedAddressBatch,
    GeocodeResult,
    RouteRequest,
    RouteResponse,
    VehicleProfile,
    VehicleType,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RouteFit API", version="0.3.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/presets")
def get_presets() -> dict[VehicleType, VehicleProfile]:
    return VEHICLE_PRESETS


@app.get("/api/geocode", response_model=list[GeocodeResult])
async def get_geocode(
    q: Optional[str] = None, query: Optional[str] = None
) -> list[GeocodeResult]:
    text = q if q not in (None, "") else query
    try:
        return await geocode(text or "")
    except GeocodingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/route", response_model=RouteResponse)
async def post_route(request: RouteRequest) -> RouteResponse:
    try:
        return await calculate_route(request)
    except RoutingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/route/optimize", response_model=RouteResponse)
async def post_route_optimize(request: RouteRequest) -> RouteResponse:
    try:
        return await optimize_route(request)
    except RoutingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.post("/api/extract-address-from-image", response_model=ExtractedAddress)
async def post_extract_address(
    file: UploadFile = File(...),
) -> ExtractedAddress:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルを送信してください")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="画像データが空です")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="画像サイズが大きすぎます（10MB以下）")
    try:
        return ExtractedAddress(**await extract_address_point(
            image_bytes, file.content_type
        ))
    except AiServiceError as exc:
        message = str(exc)
        status = 503 if "GEMINI_API_KEY" in message else 502
        raise HTTPException(status_code=status, detail=message) from exc


@app.post("/api/extract-addresses-from-images", response_model=ExtractedAddressBatch)
async def post_extract_addresses(
    files: List[UploadFile] = File(...),
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key"),
) -> ExtractedAddressBatch:
    if not files:
        raise HTTPException(status_code=400, detail="画像ファイルを送信してください")
    images: List[tuple[bytes, str]] = []
    for upload in files:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="画像ファイルを送信してください")
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail="画像データが空です")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="画像サイズが大きすぎます（10MB以下）")
        images.append((data, upload.content_type))
    try:
        payload = await extract_address_points_from_images(images, x_gemini_api_key)
    except AiServiceError as exc:
        message = str(exc)
        status = 503 if "GEMINI_API_KEY" in message else 502
        raise HTTPException(status_code=status, detail=message) from exc
    return ExtractedAddressBatch(**payload)


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
