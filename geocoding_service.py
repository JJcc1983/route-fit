import asyncio
import re
import unicodedata
from typing import List

import httpx

from schemas import GeocodeResult

GSI_ADDRESS_SEARCH_URL = (
    "https://msearch.gsi.go.jp/address-search/AddressSearch"
)
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_USER_AGENT = "RouteFit-Navigation/1.0"
GEOCODE_TIMEOUT_SEC = 20.0
GEOCODE_MAX_RETRIES = 3
GEOCODE_RETRY_BACKOFF_SEC = 0.5
GSI_MAX_RESULTS = 5
NOMINATIM_MAX_RESULTS = 5

HYPHEN_VARIANTS = "\u2212\u2010\u2011\u2012\u2013\u2014\u2015\u2500\u2501"
KANJI_SIMPLE_DIGITS = "〇一二三四五六七八九"
KANJI_UNITS = "十百千万"
KANJI_DIGITS = KANJI_SIMPLE_DIGITS + KANJI_UNITS
KANJI_ADDRESS_PATTERN = re.compile(
    r"(?:[" + KANJI_DIGITS + r"]+(?=丁目|番|号)|(?<=丁目)[" + KANJI_DIGITS + r"]+)"
)


class GeocodingError(RuntimeError):
    pass


def normalize_query(query: str) -> str:
    if not query:
        return ""
    text = unicodedata.normalize("NFKC", str(query))
    for variant in HYPHEN_VARIANTS:
        text = text.replace(variant, "-")
    text = _convert_kanji_numbers(text)
    return " ".join(text.split())


def _kanji_numeral_to_int(s: str):
    units = "\u5341\u767e\u5343\u4e07"
    if not any(ch in units for ch in s):
        value = 0
        for ch in s:
            if ch in KANJI_SIMPLE_DIGITS:
                value = value * 10 + KANJI_SIMPLE_DIGITS.index(ch)
            else:
                return None
        return value
    total = 0
    current = 0
    for ch in s:
        if ch in KANJI_SIMPLE_DIGITS:
            current = KANJI_SIMPLE_DIGITS.index(ch)
        elif ch == "\u5341":
            if current == 0:
                current = 1
            total += current * 10
            current = 0
        elif ch == "\u767e":
            if current == 0:
                current = 1
            total += current * 100
            current = 0
        elif ch == "\u5343":
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        elif ch == "\u4e07":
            total = (total + current) * 10000
            current = 0
        else:
            return None
    return total + current


def _convert_kanji_numbers(text: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        digits = match.group(0)
        if all(ch in KANJI_DIGITS for ch in digits):
            value = _kanji_numeral_to_int(digits)
            if value is not None:
                return str(value)
        return match.group(0)

    return KANJI_ADDRESS_PATTERN.sub(replace, text)


def _to_results(
    items: List[dict], display_key: str, coords_mode: str
) -> List[GeocodeResult]:
    results: List[GeocodeResult] = []
    for item in items:
        if coords_mode == "lnglat":
            coordinates = (
                item.get("geometry", {}).get("coordinates") or []
            )
            if len(coordinates) < 2:
                continue
            lng, lat = coordinates[0], coordinates[1]
            title = item.get("properties", {}).get(display_key)
        else:
            if item.get("lat") is None or item.get("lon") is None:
                continue
            lat, lng = item.get("lat"), item.get("lon")
            title = item.get(display_key)
        try:
            results.append(
                GeocodeResult(
                    display_name=str(title or "名称不明"),
                    lat=float(lat),
                    lng=float(lng),
                )
            )
        except (TypeError, ValueError):
            continue
    return results


async def _get_json(url: str, params: dict, error_message: str) -> list:
    payload = None
    last_error: Exception | None = None
    for attempt in range(GEOCODE_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=GEOCODE_TIMEOUT_SEC, follow_redirects=True
            ) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": GEOCODE_USER_AGENT},
                )
                response.raise_for_status()
            payload = response.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < GEOCODE_MAX_RETRIES:
                await asyncio.sleep(GEOCODE_RETRY_BACKOFF_SEC * (attempt + 1))
    if payload is None:
        raise GeocodingError(f"{error_message}: {last_error}")
    if not isinstance(payload, list):
        return []
    return payload


async def _gsi_search(query: str, limit: int) -> List[GeocodeResult]:
    try:
        payload = await _get_json(
            GSI_ADDRESS_SEARCH_URL,
            {"q": query},
            "国土地理院APIへの問い合わせに失敗",
        )
    except GeocodingError:
        return []
    return _to_results(payload[:limit], display_key="title", coords_mode="lnglat")


async def _nominatim_search(query: str, limit: int) -> List[GeocodeResult]:
    try:
        payload = await _get_json(
            NOMINATIM_SEARCH_URL,
            {
                "q": query,
                "format": "json",
                "countrycodes": "jp",
                "limit": limit,
            },
            "ジオコーディングに失敗しました",
        )
    except GeocodingError:
        return []
    return _to_results(
        payload[:limit], display_key="display_name", coords_mode="latlon"
    )


def _prefer_exact_matches(
    results: List[GeocodeResult], query: str
) -> List[GeocodeResult]:
    key = query.strip()
    if not key or not results:
        return results
    return sorted(
        results,
        key=lambda result: (
            0 if key in result.display_name or result.display_name.startswith(key) else 1
        ),
    )


async def geocode(query: str) -> List[GeocodeResult]:
    normalized = normalize_query(query)
    if not normalized:
        return []
    try:
        results = await _gsi_search(normalized, GSI_MAX_RESULTS)
    except GeocodingError:
        results = []
    if results:
        return _prefer_exact_matches(results, normalized)
    nominatim = await _nominatim_search(normalized, NOMINATIM_MAX_RESULTS)
    return _prefer_exact_matches(nominatim, normalized)
