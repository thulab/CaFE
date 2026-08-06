from __future__ import annotations

from dataclasses import dataclass


FEV_DATASET_REPOSITORY = "autogluon/fev_datasets"
FEV_DATASET_REVISION = "f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8"
FEV_TASK_REPOSITORY = "autogluon/fev"
FEV_TASK_REVISION = "6796a0c031e2fa99667ad836c9cf7e2d2c5b2112"
FEV_TASKS_SHA256 = (
    "c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9"
)


@dataclass(frozen=True)
class FevBenchConfig:
    config_id: str
    source_path: str
    frequency: str
    target_columns: tuple[str, ...]
    known_dynamic_columns: tuple[str, ...] = ()
    past_dynamic_columns: tuple[str, ...] = ()
    static_columns: tuple[str, ...] = ()
    categorical_dynamic_levels: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    sha256: str = ""
    size_bytes: int = 0

    @property
    def parquet_name(self) -> str:
        return self.source_path.rsplit("/", 1)[-1]


FEV_BENCH_CONFIGS = {
    config.config_id: config
    for config in (
        FevBenchConfig(
            config_id="ETT_1H",
            source_path="ETT/1H/train-00000-of-00001.parquet",
            frequency="h",
            target_columns=(
                "HUFL",
                "HULL",
                "MUFL",
                "MULL",
                "LUFL",
                "LULL",
                "OT",
            ),
            sha256=(
                "e742c83a9eb6af84c2c4f8105193dbdde3cfc729d25be5985ffed96aae78581c"
            ),
            size_bytes=531_145,
        ),
        FevBenchConfig(
            config_id="jena_weather_1H",
            source_path="jena_weather/1H/train-00000-of-00001.parquet",
            frequency="h",
            target_columns=tuple(f"target_{index}" for index in range(21)),
            sha256=(
                "3fb5a84924facab11a47ffab0d0cc96136e6a70cadb09c36cf89c364bf2c1d86"
            ),
            size_bytes=765_995,
        ),
        FevBenchConfig(
            config_id="boomlet_1282",
            source_path="boomlet/1282/train-00000-of-00001.parquet",
            frequency="min",
            target_columns=tuple(f"target_{index}" for index in range(35)),
            sha256=(
                "3816a03407762f99e8f11890385b6f2afa9ae50016185a851b8750f289e0734c"
            ),
            size_bytes=1_385_189,
        ),
        FevBenchConfig(
            config_id="uci_air_quality_1H",
            source_path="uci_air_quality/1H/train-00000-of-00001.parquet",
            frequency="h",
            target_columns=("CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"),
            known_dynamic_columns=("T", "RH", "AH"),
            sha256=(
                "a851746a468adaad941201ac0dfcef1a94c62593458380a435f1f2c3db406b34"
            ),
            size_bytes=287_093,
        ),
        FevBenchConfig(
            config_id="solar_with_weather_1H",
            source_path=(
                "solar_with_weather/1H/train-00000-of-00001.parquet"
            ),
            frequency="h",
            target_columns=("target",),
            known_dynamic_columns=(
                "wind_speed",
                "day_length",
                "humidity",
                "rain_1h",
                "snow_1h",
                "temp",
                "pressure",
            ),
            past_dynamic_columns=(
                "global_horizontal_irradiance",
                "clouds_all",
            ),
            sha256=(
                "b869eca73035a6e480798210ad93ba114e172e039605101e192884778f1d8a77"
            ),
            size_bytes=830_505,
        ),
        FevBenchConfig(
            config_id="proenfo_gfc14",
            source_path="proenfo_gfc14/train-00000-of-00001.parquet",
            frequency="h",
            target_columns=("target",),
            known_dynamic_columns=("airtemperature",),
            sha256=(
                "384505830c3b16a732c5feddfa41195e06387b3908a9a9fd78116298e8c6ecda"
            ),
            size_bytes=200_402,
        ),
        FevBenchConfig(
            config_id="rohlik_orders_1D",
            source_path="rohlik_orders/1D/train-00000-of-00001.parquet",
            frequency="D",
            target_columns=("orders",),
            known_dynamic_columns=(
                "holiday",
                "shops_closed",
                "winter_school_holidays",
                "school_holidays",
            ),
            past_dynamic_columns=(
                "shutdown",
                "mini_shutdown",
                "blackout",
                "mov_change",
                "frankfurt_shutdown",
                "precipitation",
                "snow",
                "user_activity_1",
                "user_activity_2",
            ),
            sha256=(
                "de43786b272ca6f36a0f5f298d2fe14277f43d1e7b595d85fb5cff0b9efbfe00"
            ),
            size_bytes=125_366,
        ),
        FevBenchConfig(
            config_id="rossmann_1D",
            source_path="rossmann/1D/train-00000-of-00001.parquet",
            frequency="D",
            target_columns=("Sales",),
            known_dynamic_columns=(
                "SchoolHoliday",
                "Promo",
                "DayOfWeek",
                "Open",
                "StateHoliday",
            ),
            past_dynamic_columns=("Customers",),
            static_columns=(
                "Store",
                "StoreType",
                "Assortment",
                "CompetitionDistance",
                "CompetitionOpenSinceMonth",
                "CompetitionOpenSinceYear",
                "Promo2",
                "Promo2SinceWeek",
                "Promo2SinceYear",
                "PromoInterval",
            ),
            categorical_dynamic_levels=(
                ("StateHoliday", ("0", "a", "b", "c")),
            ),
            sha256=(
                "ab6c84fb0975405cbfb3caa14cd03a8a1beb9508bcffc0b3c6ce99e78f64ca3a"
            ),
            size_bytes=3_864_893,
        ),
        FevBenchConfig(
            config_id="hospital_admissions_1D",
            source_path=(
                "hospital_admissions/1D/train-00000-of-00001.parquet"
            ),
            frequency="D",
            target_columns=("target",),
            sha256=(
                "a9eea1e54e2018bad8f6dd9ee5a6ebdcf561f06b1900e634ea2022564e09b3c5"
            ),
            size_bytes=40_418,
        ),
    )
}
