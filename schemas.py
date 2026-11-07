from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class VehicleType(str, Enum):
    MOPED_50 = "moped_50"  # 原付一種 (〜50cc)
    SCOOTER_125 = "scooter_125"  # 原付二種 (51〜125cc)
    MOTORCYCLE = "motorcycle"  # 中型・大型二輪 (126cc〜)
    KEI_TRUCK = "kei_truck"  # 軽トラ・軽バン
    TRUCK_2T = "truck_2t"  # 2tトラック (標準/ロング)
    TRUCK_4T = "truck_4t"  # 4t中型トラック
    CUSTOM = "custom"  # ユーザー独自寸法


class VehicleProfile(BaseModel):
    type: VehicleType
    name: str = Field(..., description="表示名")
    height_m: float = Field(..., description="全高（m）")
    width_m: float = Field(..., description="全幅（m）")
    length_m: float = Field(..., description="全長（m）")
    weight_t: float = Field(..., description="車両総重量（t）")
    engine_cc: Optional[int] = Field(None, description="排気量（cc）")
    max_speed_city_kmh: float = Field(
        60.0, description="一般道での最高許容速度（km/h）"
    )
    max_speed_highway_kmh: float = Field(
        100.0, description="高速での最高許容速度（km/h）"
    )
    allow_motorway: bool = Field(False, description="高速道路許可")
    allow_trunk_bypass: bool = Field(False, description="自動車専用バイパス許可")
    require_hook_turn: bool = Field(False, description="二段階右折適用")
    height_margin_m: float = Field(0.2, description="高さ安全マージン（m）")
    avoid_underpass: bool = Field(False, description="アンダーパス原則回避")


class LocationPoint(BaseModel):
    lat: float
    lng: float
    name: Optional[str] = None


class RouteRequest(BaseModel):
    origin: LocationPoint
    destination: LocationPoint
    vehicle_profile: VehicleProfile
    waypoints: list[LocationPoint] = Field(
        default_factory=list, description="経由地（配送先・立ち寄り地）リスト"
    )
    departure_time: Optional[str] = None


class RestrictionAlert(BaseModel):
    alert_type: str
    title: str
    description: str
    lat: float
    lng: float


class GeocodeResult(BaseModel):
    display_name: str
    lat: float
    lng: float


class ExtractedAddress(BaseModel):
    address: str = ""
    model_used: str = ""
    matches: List[GeocodeResult] = Field(default_factory=list)


class ExtractedAddressPoint(BaseModel):
    address: str
    lat: float
    lng: float
    display_name: str


class ExtractedAddressBatch(BaseModel):
    addresses: List[str] = Field(default_factory=list)
    points: List[ExtractedAddressPoint] = Field(default_factory=list)
    model_used: str = ""
    resolved_count: int = 0


class RouteResponse(BaseModel):
    geometry: dict
    distance_km: float
    duration_min: float
    alerts: List[RestrictionAlert]
    profile_applied: VehicleProfile
    ordered_waypoints: List[LocationPoint] = Field(
        default_factory=list, description="巡回最適化後の経由地順序"
    )
    navigation_steps: List[dict] = Field(
        default_factory=list, description="ターンバイターン案内ステップ"
    )


VEHICLE_PRESETS: dict[VehicleType, VehicleProfile] = {
    VehicleType.MOPED_50: VehicleProfile(
        type=VehicleType.MOPED_50,
        name="🛵 原付一種 (50cc)",
        height_m=1.1,
        width_m=0.7,
        length_m=1.8,
        weight_t=0.1,
        engine_cc=50,
        allow_motorway=False,
        allow_trunk_bypass=False,
        require_hook_turn=True,
        avoid_underpass=True,
        max_speed_city_kmh=30.0,
        max_speed_highway_kmh=30.0,
    ),
    VehicleType.SCOOTER_125: VehicleProfile(
        type=VehicleType.SCOOTER_125,
        name="🏍️ 原付二種 (125cc)",
        height_m=1.1,
        width_m=0.75,
        length_m=1.9,
        weight_t=0.15,
        engine_cc=125,
        allow_motorway=False,
        allow_trunk_bypass=True,
        require_hook_turn=False,
        avoid_underpass=False,
        max_speed_city_kmh=60.0,
        max_speed_highway_kmh=60.0,
    ),
    VehicleType.MOTORCYCLE: VehicleProfile(
        type=VehicleType.MOTORCYCLE,
        name="🏍️ 中・大型二輪",
        height_m=1.2,
        width_m=0.85,
        length_m=2.1,
        weight_t=0.25,
        engine_cc=400,
        allow_motorway=True,
        allow_trunk_bypass=True,
        require_hook_turn=False,
        avoid_underpass=False,
    ),
    VehicleType.KEI_TRUCK: VehicleProfile(
        type=VehicleType.KEI_TRUCK,
        name="🚙 軽トラ・軽バン",
        height_m=1.95,
        width_m=1.47,
        length_m=3.39,
        weight_t=1.5,
        allow_motorway=True,
        allow_trunk_bypass=True,
        require_hook_turn=False,
        avoid_underpass=False,
    ),
    VehicleType.TRUCK_2T: VehicleProfile(
        type=VehicleType.TRUCK_2T,
        name="🚚 2tアルミバン",
        height_m=3.0,
        width_m=1.89,
        length_m=4.69,
        weight_t=4.5,
        allow_motorway=True,
        allow_trunk_bypass=True,
        require_hook_turn=False,
        avoid_underpass=False,
    ),
    VehicleType.TRUCK_4T: VehicleProfile(
        type=VehicleType.TRUCK_4T,
        name="🚚 4tウィング",
        height_m=3.5,
        width_m=2.35,
        length_m=8.45,
        weight_t=7.9,
        allow_motorway=True,
        allow_trunk_bypass=True,
        require_hook_turn=False,
        avoid_underpass=False,
        max_speed_city_kmh=60.0,
        max_speed_highway_kmh=80.0,
    ),
}