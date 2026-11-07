import asyncio
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx

from schemas import (
    LocationPoint,
    RestrictionAlert,
    RouteRequest,
    RouteResponse,
    VehicleProfile,
)

OSRM_DRIVING_URL = "https://router.project-osrm.org/route/v1/driving"
OSRM_TRIP_URL = "https://router.project-osrm.org/trip/v1/driving"
ROUTE_TIMEOUT_SEC = 60.0
OSRM_MAX_RETRIES = 3
OSRM_RETRY_BACKOFF_SEC = 1.0
GUARD_MATCH_THRESHOLD_M = 80.0
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LNG = 111320.0
CITY_CRUISE_KMH = 40.0
CITY_SPEED_FACTOR = 0.75


class RoutingError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetourWaypoint:
    lat: float
    lng: float
    name: str

    def as_location_point(self) -> LocationPoint:
        return LocationPoint(lat=self.lat, lng=self.lng, name=self.name)


@dataclass(frozen=True)
class RestrictionSpot:
    id: str
    name: str
    lat: float
    lng: float
    max_height_m: float
    max_weight_t: Optional[float] = None
    detour: Optional[DetourWaypoint] = None


@dataclass(frozen=True)
class TwoWheelRestrictionSpot:
    id: str
    name: str
    lat: float
    lng: float
    min_engine_cc: int
    detour: DetourWaypoint


KNOWN_RESTRICTIONS: List[RestrictionSpot] = [
    RestrictionSpot(
        id="nakatsu-guard",
        name="中津近郊 高架下ガード",
        lat=34.70526,
        lng=135.49712,
        max_height_m=2.8,
        max_weight_t=None,
        detour=DetourWaypoint(
            lat=34.70700,
            lng=135.49950,
            name="東側バイパス・天神橋筋方面へ迂回",
        ),
    ),
    RestrictionSpot(
        id="nishi-yokobori-central",
        name="西横堀筋 中央大通高架下",
        lat=34.6818,
        lng=135.4985,
        max_height_m=2.0,
        max_weight_t=2.0,
        detour=DetourWaypoint(
            lat=34.6818,
            lng=135.4960,
            name="四つ橋筋側へ迂回",
        ),
    ),
    RestrictionSpot(
        id="deiribashi-kita",
        name="出入橋北側 高架下連絡路",
        lat=34.6970,
        lng=135.4900,
        max_height_m=2.1,
        max_weight_t=2.0,
        detour=DetourWaypoint(
            lat=34.6960,
            lng=135.4880,
            name="なにわ筋側へ迂回",
        ),
    ),
]


TWO_WHEEL_RESTRICTION_SPOTS: List[TwoWheelRestrictionSpot] = [
    TwoWheelRestrictionSpot(
        id="shin-midosuji-mainline",
        name="新御堂筋連絡 本線高架部（梅田〜中津間）",
        lat=34.70563,
        lng=135.50092,
        min_engine_cc=51,
        detour=DetourWaypoint(
            lat=34.70550,
            lng=135.49500,
            name="平面側道（御堂筋側道・十三筋方面）へ迂回",
        ),
    ),
]


def _segment_distance_m(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _spot_relative_to_route_m(
    lat: float, lng: float, coordinates: List[List[float]]
) -> Tuple[float, float]:
    ref_cos = math.cos(math.radians(lat))

    def project(pt_lng: float, pt_lat: float) -> Tuple[float, float]:
        return (
            (pt_lng - lng) * M_PER_DEG_LNG * ref_cos,
            (pt_lat - lat) * M_PER_DEG_LAT,
        )

    best_distance = float("inf")
    best_progress = 0.0
    cumulative = 0.0
    prev_progress = 0.0
    prev: Tuple[float, float] = project(coordinates[0][0], coordinates[0][1])
    for coord in coordinates[1:]:
        current = project(coord[0], coord[1])
        seg_len = math.hypot(current[0] - prev[0], current[1] - prev[1])
        distance = _segment_distance_m(
            0.0, 0.0, prev[0], prev[1], current[0], current[1]
        )
        if distance < best_distance:
            best_distance = distance
            best_progress = prev_progress
        cumulative += seg_len
        prev_progress = cumulative
        prev = current
    return best_distance, best_progress


def _route_hits_spot(
    geometry: dict, lat: float, lng: float
) -> Tuple[bool, float]:
    coordinates = geometry.get("coordinates") or []
    if len(coordinates) < 2:
        return False, 0.0
    distance, progress = _spot_relative_to_route_m(lat, lng, coordinates)
    return distance <= GUARD_MATCH_THRESHOLD_M, progress


def _coord_str(waypoints: List[LocationPoint]) -> str:
    return ";".join(f"{p.lng},{p.lat}" for p in waypoints)


async def _osrm_get(url: str, params: dict) -> dict:
    payload = None
    last_error: Exception | None = None
    for attempt in range(OSRM_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=ROUTE_TIMEOUT_SEC, follow_redirects=True
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
            payload = response.json()
            break
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < OSRM_MAX_RETRIES:
                await asyncio.sleep(OSRM_RETRY_BACKOFF_SEC * (attempt + 1))
    if payload is None:
        raise RoutingError(
            f"OSRMへのリクエストに失敗しました: {last_error}"
        ) from last_error
    if payload.get("code") != "Ok":
        message = payload.get("message", payload.get("code"))
        raise RoutingError(f"OSRMが結果を返しませんでした: {message}")
    return payload


OSRM_BASE_PARAMS = {
    "overview": "full",
    "geometries": "geojson",
    "steps": "true",
}


def _extract_navigation_steps(route: dict) -> List[dict]:
    steps: List[dict] = []
    for leg_index, leg in enumerate(route.get("legs") or []):
        for step in leg.get("steps") or []:
            maneuver = step.get("maneuver") or {}
            location = maneuver.get("location") or []
            if len(location) < 2:
                continue
            steps.append(
                {
                    "leg_index": leg_index,
                    "type": maneuver.get("type"),
                    "modifier": maneuver.get("modifier"),
                    "name": step.get("name") or "",
                    "distance": float(step.get("distance") or 0.0),
                    "duration": float(step.get("duration") or 0.0),
                    "lat": float(location[1]),
                    "lng": float(location[0]),
                }
            )
    return steps


async def _fetch_osrm_route(waypoints: List[LocationPoint]) -> dict:
    params = dict(OSRM_BASE_PARAMS)
    payload = await _osrm_get(f"{OSRM_DRIVING_URL}/{_coord_str(waypoints)}", params)
    routes = payload.get("routes") or []
    if not routes:
        raise RoutingError("OSRMが経路を返しませんでした")
    route = routes[0]
    geometry = route.get("geometry")
    if not geometry or not geometry.get("coordinates"):
        raise RoutingError("OSRMの応答にgeometryが含まれていません")
    return {
        "geometry": geometry,
        "distance_km": route["distance"] / 1000.0,
        "duration_min": route["duration"] / 60.0,
        "navigation_steps": _extract_navigation_steps(route),
    }


async def _fetch_osrm_trip(
    stops: List[LocationPoint],
) -> Tuple[List[LocationPoint], dict]:
    params = {
        "source": "first",
        "destination": "last",
        "roundtrip": "false",
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
    }
    payload = await _osrm_get(f"{OSRM_TRIP_URL}/{_coord_str(stops)}", params)
    trips = payload.get("trips") or []
    waypoint_indices = payload.get("waypoints") or []
    if not trips or len(waypoint_indices) != len(stops):
        raise RoutingError("OSRM Trip APIの応答を解釈できませんでした")
    order = sorted(
        range(len(stops)),
        key=lambda index: waypoint_indices[index].get("waypoint_index", index),
    )
    ordered_stops = [stops[index] for index in order]
    return ordered_stops, trips[0]


def estimate_duration_min(distance_km: float, vehicle: VehicleProfile) -> float:
    city_cap_kmh = min(CITY_CRUISE_KMH, vehicle.max_speed_city_kmh)
    speed_cap_kmh = min(city_cap_kmh, vehicle.max_speed_highway_kmh)
    effective_kmh = speed_cap_kmh * CITY_SPEED_FACTOR
    if effective_kmh <= 0:
        effective_kmh = 1.0
    return distance_km / effective_kmh * 60.0


def _spot_exceeds(
    spot: RestrictionSpot, vehicle: VehicleProfile
) -> Tuple[bool, bool]:
    effective_height = vehicle.height_m + vehicle.height_margin_m
    height_over = effective_height > spot.max_height_m
    weight_over = (
        spot.max_weight_t is not None
        and vehicle.weight_t > spot.max_weight_t
    )
    return height_over, weight_over


def _restriction_alerts(
    triggered: List[RestrictionSpot], vehicle: VehicleProfile
) -> List[RestrictionAlert]:
    effective_height = vehicle.height_m + vehicle.height_margin_m
    alerts: List[RestrictionAlert] = []
    for spot in triggered:
        height_over, weight_over = _spot_exceeds(spot, vehicle)
        detour_name = spot.detour.name if spot.detour else "側道へ迂回"
        if height_over and weight_over:
            title = (
                f"高さ{spot.max_height_m:.1f}m・重量{spot.max_weight_t:.1f}t "
                "制限を迂回"
            )
            description = (
                f"{spot.name}は高さ制限{spot.max_height_m:.1f}m・"
                f"重量制限{spot.max_weight_t:.1f}t。車両全高{vehicle.height_m}m"
                f"+安全マージン{vehicle.height_margin_m}m={effective_height:.2f}m"
                f"・重量{vehicle.weight_t:.1f}tのため、{detour_name}します。"
            )
        elif weight_over:
            title = f"重量制限 {spot.max_weight_t:.1f}t 高架下を迂回"
            description = (
                f"{spot.name}は重量制限{spot.max_weight_t:.1f}t。"
                f"車両重量{vehicle.weight_t:.1f}tのため、{detour_name}します。"
            )
        else:
            title = f"高さ制限 {spot.max_height_m:.1f}m ガード下を迂回"
            description = (
                f"{spot.name}は高さ制限{spot.max_height_m:.1f}m。"
                f"車両全高{vehicle.height_m}m+安全マージン{vehicle.height_margin_m}m"
                f"={effective_height:.2f}mのため、{detour_name}します。"
            )
        alerts.append(
            RestrictionAlert(
                alert_type="height_restriction",
                title=title,
                description=description,
                lat=spot.lat,
                lng=spot.lng,
            )
        )
    return alerts


def _two_wheel_alerts(
    triggered: List[TwoWheelRestrictionSpot], vehicle: VehicleProfile
) -> List[RestrictionAlert]:
    return [
        RestrictionAlert(
            alert_type="MOPED_BAN_AVOIDED",
            title="原付通行禁止（新御堂筋本線）を迂回",
            description="50cc以下は本線高架を通行できないため、平面側道へ迂回します",
            lat=spot.lat,
            lng=spot.lng,
        )
        for spot in triggered
    ]


def _is_two_wheel_banned(
    vehicle: VehicleProfile, spot: TwoWheelRestrictionSpot
) -> bool:
    if vehicle.engine_cc is not None and vehicle.engine_cc < spot.min_engine_cc:
        return True
    return not vehicle.allow_trunk_bypass


def _build_stop_list(request: RouteRequest) -> List[LocationPoint]:
    return [request.origin] + list(request.waypoints) + [request.destination]


def _stop_positions_m(
    geometry_coordinates: List[List[float]], stops: List[LocationPoint]
) -> List[float]:
    positions: List[float] = []
    for stop in stops:
        _, progress = _spot_relative_to_route_m(
            stop.lat, stop.lng, geometry_coordinates
        )
        positions.append(progress)
    return positions


def _insert_detours(
    stops: List[LocationPoint],
    stop_positions: List[float],
    triggered: List[Tuple[RestrictionSpot, float]],
) -> List[LocationPoint]:
    insertions: dict[int, List[LocationPoint]] = {}
    for spot, progress in triggered:
        if spot.detour is None:
            continue
        after_index = 0
        for index, position in enumerate(stop_positions):
            if position <= progress:
                after_index = index + 1
        insertions.setdefault(after_index, []).append(
            spot.detour.as_location_point()
        )
    ordered: List[LocationPoint] = []
    for index, stop in enumerate(stops):
        ordered.append(stop)
        ordered.extend(insertions.get(index + 1, []))
    ordered.extend(insertions.get(0, []))
    return ordered


async def optimize_route(request: RouteRequest) -> RouteResponse:
    if len(request.waypoints) < 2:
        response = await calculate_route(request)
        return response.model_copy(update={"ordered_waypoints": []})

    stops = _build_stop_list(request)
    ordered_stops, _ = await _fetch_osrm_trip(stops)
    optimized_request = RouteRequest(
        origin=ordered_stops[0],
        waypoints=ordered_stops[1:-1],
        destination=ordered_stops[-1],
        vehicle_profile=request.vehicle_profile,
        departure_time=request.departure_time,
    )
    response = await calculate_route(optimized_request)
    return response.model_copy(
        update={"ordered_waypoints": ordered_stops[1:-1]}
    )


async def calculate_route(request: RouteRequest) -> RouteResponse:
    vehicle = request.vehicle_profile

    stops = _build_stop_list(request)
    direct_route = await _fetch_osrm_route(stops)
    stop_positions = _stop_positions_m(
        direct_route["geometry"]["coordinates"], stops
    )

    restriction_triggered: List[Tuple[RestrictionSpot, float]] = []
    for spot in KNOWN_RESTRICTIONS:
        height_over, weight_over = _spot_exceeds(spot, vehicle)
        if not (height_over or weight_over):
            continue
        hits, progress = _route_hits_spot(
            direct_route["geometry"], spot.lat, spot.lng
        )
        if hits:
            restriction_triggered.append((spot, progress))

    two_wheel_triggered: List[Tuple[TwoWheelRestrictionSpot, float]] = []
    for spot in TWO_WHEEL_RESTRICTION_SPOTS:
        if not _is_two_wheel_banned(vehicle, spot):
            continue
        hits, progress = _route_hits_spot(
            direct_route["geometry"], spot.lat, spot.lng
        )
        if hits:
            two_wheel_triggered.append((spot, progress))

    triggered = sorted(
        restriction_triggered + two_wheel_triggered, key=lambda item: item[1]
    )

    route = direct_route
    if triggered:
        waypoints = _insert_detours(stops, stop_positions, triggered)
        route = await _fetch_osrm_route(waypoints)

    return RouteResponse(
        geometry=route["geometry"],
        distance_km=route["distance_km"],
        duration_min=estimate_duration_min(route["distance_km"], vehicle),
        alerts=(
            _restriction_alerts(
                [spot for spot, _ in restriction_triggered], vehicle
            )
            + _two_wheel_alerts(
                [spot for spot, _ in two_wheel_triggered], vehicle
            )
        ),
        profile_applied=vehicle,
        navigation_steps=route.get("navigation_steps") or [],
    )
