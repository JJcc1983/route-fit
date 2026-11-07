import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import dotenv_values

from schemas import GeocodeResult

GEMINI_MODEL_PRIMARY = "gemini-3.5-flash-lite"
GEMINI_MODEL_FALLBACKS = (
    "gemini-2.5-flash-lite",
    "gemini-3.7-flash",
)

ADDRESS_EXTRACTION_PROMPT = (
    "与えられた伝票やメモの画像から、お届け先（配送先）の住所文字列のみを抽出して"
    'JSON形式 {"address": "..."} で返してください。'
    "番地や号、建物名・部屋番号も含めて正確に抽出してください。"
    "住所が読み取れない場合は {\"address\": \"\"} を返してください。"
)
ADDRESSES_EXTRACTION_PROMPT = (
    "与えられた画像（または複数画像）から、見つかったすべての配送先住所を抽出してください。"
    "1枚の画像に複数の伝票やメモが写っている場合も、すべて漏れなく抽出して"
    ' JSON形式 {"addresses": ["住所1", "住所2", ...]} で返してください。'
    "番地や号、建物名・部屋番号も含めて正確に抽出してください。"
    "住所が1件も読み取れない場合は {\"addresses\": []} を返してください。"
)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class AiServiceError(RuntimeError):
    pass


def _load_env_file() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            settings = dotenv_values(env_path)
        except Exception:
            return
        key = (settings.get("GEMINI_API_KEY") or "").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key


_load_env_file()


def _parse_addresses_json(text: str) -> List[str]:
    if not text:
        raise AiServiceError("AI応答が空です")
    candidate = text.strip()
    block = JSON_BLOCK_PATTERN.search(candidate)
    if block:
        candidate = block.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AiServiceError("AI応答をJSONとして解釈できませんでした") from exc
    addresses = payload.get("addresses") if isinstance(payload, dict) else None
    if addresses is None:
        raise AiServiceError("AI応答に addresses フィールドがありません")
    if not isinstance(addresses, list):
        raise AiServiceError("addresses が配列ではありません")
    return [
        str(address).strip() for address in addresses if str(address).strip()
    ]


def _call_gemini_multi(
    client, model: str, images: List[Tuple[bytes, str]], prompt: str
) -> str:
    from google.genai import types

    contents = [prompt]
    for image_bytes, mime_type in images:
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )
    response = client.models.generate_content(model=model, contents=contents)
    return getattr(response, "text", "") or ""


async def extract_addresses_from_images(
    images: List[Tuple[bytes, str]]
) -> Tuple[List[str], str]:
    if not images:
        raise AiServiceError("画像が送信されていません")
    client = _build_client()
    last_error: Optional[Exception] = None
    for model in (GEMINI_MODEL_PRIMARY, *GEMINI_MODEL_FALLBACKS):
        try:
            text = await asyncio.to_thread(
                _call_gemini_multi,
                client,
                model,
                images,
                ADDRESSES_EXTRACTION_PROMPT,
            )
            return _parse_addresses_json(text), model
        except AiServiceError:
            raise
        except Exception as exc:
            last_error = exc
            continue
    raise AiServiceError(f"住所抽出に失敗しました: {last_error}")


async def extract_address_points_from_images(
    images: List[Tuple[bytes, str]], api_key: Optional[str] = None
) -> dict:
    from geocoding_service import geocode

    addresses, model_used = await extract_addresses_from_images(images, api_key)
    points: List[dict] = []
    resolved: List[str] = []
    for address in addresses:
        matches = await geocode(address)
        if not matches:
            continue
        best = matches[0]
        resolved.append(address)
        points.append(
            {
                "address": address,
                "lat": best.lat,
                "lng": best.lng,
                "display_name": best.display_name,
            }
        )
    return {
        "addresses": addresses,
        "points": points,
        "model_used": model_used,
        "resolved_count": len(points),
    }


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise AiServiceError(
            "GEMINI_API_KEY が未設定です（環境変数にAPIキーを設定してください）"
        )
    return key


def _build_client():
    try:
        from google import genai
    except ImportError as exc:
        raise AiServiceError("google-genai SDK が未インストールです") from exc
    return genai.Client(api_key=_get_api_key())


def _parse_address_json(text: str) -> str:
    if not text:
        raise AiServiceError("AI応答が空です")
    candidate = text.strip()
    block = JSON_BLOCK_PATTERN.search(candidate)
    if block:
        candidate = block.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AiServiceError("AI応答をJSONとして解釈できませんでした") from exc
    address = payload.get("address") if isinstance(payload, dict) else None
    if address is None:
        raise AiServiceError("AI応答に address フィールドがありません")
    return str(address).strip()


def _call_gemini(client, model: str, image_bytes: bytes, mime_type: str) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[
            ADDRESS_EXTRACTION_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return getattr(response, "text", "") or ""


async def extract_address_from_image(
    image_bytes: bytes, mime_type: str, api_key: Optional[str] = None
) -> Tuple[str, str]:
    client = _build_client()
    last_error: Optional[Exception] = None
    for model in (GEMINI_MODEL_PRIMARY, *GEMINI_MODEL_FALLBACKS):
        try:
            text = await asyncio.to_thread(
                _call_gemini, client, model, image_bytes, mime_type
            )
            return _parse_address_json(text), model
        except AiServiceError:
            raise
        except Exception as exc:
            last_error = exc
            continue
    raise AiServiceError(f"住所抽出に失敗しました: {last_error}")


async def extract_address_point(
    image_bytes: bytes, mime_type: str, api_key: Optional[str] = None
) -> dict:
    from geocoding_service import geocode

    address, model_used = await extract_address_from_image(
        image_bytes, mime_type, api_key
    )
    matches: List[GeocodeResult] = []
    if address:
        matches = await geocode(address)
    return {
        "address": address,
        "model_used": model_used,
        "matches": matches,
    }
