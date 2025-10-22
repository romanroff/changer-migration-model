import os
from dataclasses import dataclass, field
from functools import singledispatchmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import csr_matrix, find, issparse
from shapely.geometry import LineString, Point

from townsnet.provision import ProvisionModel
from townsnet.provision.service_type import ServiceType
from changer_migration.ueqi import UEQI_GROUPS
from changer_migration.creating_map import create_anchor_flow_map, save_static_anchor_flow_png

# ---- Константы/настройки ----
OFFSET_PX: float = 300.0  # сдвиг при генерации новой точки сервиса на N метров
DEFAULT_MATRIX_DIR = Path("data/provision")  # базовый путь для матриц
UPDATED_SUBDIR = "updated"  # подкаталог с обновлёнными матрицами
SERVICE_GROUPS_PLACEHOLDER: Dict[str, List[str]] = {
    # TODO: populate with function names and related service types.
    # "example_function": ["ServiceTypeName1", "ServiceTypeName2"],
}


@dataclass
class MigrationFlowModel:
    """
    Модель потоков миграции и анализа «якорных» городов.
    """

    # --- Внешние объекты-модели ---
    provision: Optional[ProvisionModel] = field(default=None, init=False)
    acc_mx: Optional[pd.DataFrame] = field(default=None, init=False)
    service_types_df: Optional[pd.DataFrame] = field(default=None, init=False)
    normatives_df: Optional[pd.DataFrame] = field(default=None, init=False)
    service_types: Optional[Dict[int, ServiceType]] = field(default=None, init=False)
    service_types_by_name: Dict[str, ServiceType] = field(default_factory=dict, init=False)

    # --- Основные таблицы/геоданные ---
    towns: Optional[gpd.GeoDataFrame] = field(default=None, init=False)
    services: Optional[gpd.GeoDataFrame] = field(default=None, init=False)
    municipal_districts: Optional[gpd.GeoDataFrame] = field(default=None, init=False)

    # --- Справочники/инфраструктура ---
    INFRASTRUCTURE: Dict[str, List[str]] = field(default_factory=dict, init=False)
    service_results: Dict[str, gpd.GeoDataFrame] = field(default_factory=dict, init=False)
    group_provision: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    # --- Матрицы/производные данные ---
    combined_matrix: Optional[csr_matrix] = field(default=None, init=False)
    flows: Optional[gpd.GeoDataFrame] = field(default=None, init=False)
    anchor: Optional[gpd.GeoDataFrame] = field(default=None, init=False)  # города + признаки якоря
    mobility: Optional[Dict[str, pd.DataFrame]] = field(default=None, init=False)

    # --- Групповые результаты ---
    group_matrices: Dict[str, csr_matrix] = field(default_factory=dict, init=False)
    group_flows: Dict[str, gpd.GeoDataFrame] = field(default_factory=dict, init=False)
    group_mobility: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict, init=False)

    # --- JSON profiles by city ---
    city_json: Dict[int, Dict] = field(default_factory=dict, init=False)


    # --- Параметры/seed ---
    seed: Optional[int] = None  # детерминизм по желанию


    def _ensure_service_types(self) -> None:
        if self.service_types is not None and self.service_types_by_name:
            return
        if self.service_types_df is None or self.normatives_df is None:
            raise RuntimeError("service_types_df and normatives_df are required to initialize ServiceType objects")
        service_list = ServiceType.initialize_service_types(self.service_types_df, self.normatives_df)
        self.service_types = {st.id: st for st in service_list}
        self.service_types_by_name = {st.name: st for st in service_list}

    def _empty_supply_dataframe(self) -> pd.DataFrame:
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not loaded.")
        return pd.DataFrame(index=self.towns.index, data={"supply": 0.0})

    def _load_supplies_from_geojson(self, source: str | Path | gpd.GeoDataFrame) -> Dict[str, pd.DataFrame]:
        if source is None:
            return {}
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not loaded.")
        if isinstance(source, (str, Path)):
            gdf = gpd.read_file(source)
        elif isinstance(source, gpd.GeoDataFrame):
            gdf = source.copy()
        else:
            raise TypeError(f"Unsupported services source type: {type(source)!r}")
        required_columns = {"town", "service_type", "capacity"}
        if not required_columns.issubset(gdf.columns):
            missing = required_columns.difference(gdf.columns)
            raise ValueError(f"Services GeoDataFrame is missing columns: {missing}")
        gdf["town"] = gdf["town"].astype(str).str.strip()
        gdf["service_type"] = gdf["service_type"].astype(str).str.strip()
        gdf["capacity"] = pd.to_numeric(gdf["capacity"], errors="coerce").fillna(0.0)
        town_lookup = {str(name).strip().lower(): idx for idx, name in self.towns["town_name"].items()}
        supplies: Dict[str, pd.DataFrame] = {}
        for service_name, group in gdf.groupby("service_type"):
            supply_df = self._empty_supply_dataframe()
            aggregated = group.groupby("town")["capacity"].sum()
            for town_name, value in aggregated.items():
                town_idx = town_lookup.get(str(town_name).strip().lower())
                if town_idx is None:
                    continue
                supply_df.at[town_idx, "supply"] = float(supply_df.at[town_idx, "supply"]) + float(value)
            supplies[service_name] = supply_df
        return supplies

    def _ensure_anchor_frame(self, anchors: Optional[pd.DataFrame] = None) -> None:
        if self.anchor is not None:
            return
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not loaded.")
        cities = self.towns.reset_index(drop=True)[["town_name", "geometry"]].copy()
        population = self.towns["population"] if "population" in self.towns.columns else pd.Series(0, index=self.towns.index)
        cities["population"] = population.fillna(0).to_numpy()
        cities["is_anchor_settlement"] = False
        if anchors is not None and "is_anchor_settlement" in anchors.columns:
            anchor_flag = anchors["is_anchor_settlement"].astype(bool)
            anchor_flag = anchor_flag.reindex(cities.index, fill_value=False)
            cities["is_anchor_settlement"] = anchor_flag.to_numpy()
        elif "is_anchor_settlement" in self.towns.columns:
            cities["is_anchor_settlement"] = self.towns["is_anchor_settlement"].astype(bool).to_numpy()
        anchor_gdf = gpd.GeoDataFrame(cities, geometry="geometry", crs=self.towns.crs)
        anchor_gdf["city_id"] = anchor_gdf.index
        self.anchor = anchor_gdf.set_index("city_id")


    # ----------------- Конструирование/загрузка -----------------

    @classmethod
    def from_pickle(cls, path: str | Path, *, seed: Optional[int] = None) -> "MigrationFlowModel":
        """
        Фабричный метод: загрузка Region из pickle и подготовка зависимостей.
        """
        m = cls(seed=seed)
        m._load_dataset(path)
        m._init_subsystems()
        return m

    # ----------------- Индексация по ID города -----------------

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    @__getitem__.register(int)
    def _(self, town_id: int):
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not initialized.")
        if town_id not in self.towns.index:
            raise KeyError(f"Can't find town with such id: {town_id}")
        return self.towns.loc[town_id]

    # ----------------- Приватные шаги инициализации -----------------

    def _load_dataset(self, path: str | Path) -> None:
        base = Path(path)
        if base.is_file():
            base = base.parent

        towns_path = base / "towns.geojson"
        acc_path = base / "acc_mx.pickle"
        st_path = base / "service_types.pickle"
        norm_path = base / "normatives.pickle"

        if not towns_path.exists() or not acc_path.exists():
            raise FileNotFoundError(
                f"Dataset is incomplete in '{base}': expected towns.geojson and acc_mx.pickle"
            )

        self.towns = gpd.read_file(towns_path)
        if "town_name" not in self.towns.columns and "name" in self.towns.columns:
            self.towns = self.towns.rename(columns={"name": "town_name"})

        import pickle as _pkl
        with open(acc_path, "rb") as f:
            self.acc_mx = _pkl.load(f)
        if not isinstance(self.acc_mx, pd.DataFrame):
            self.acc_mx = pd.DataFrame(self.acc_mx)

        if not (self.towns.index.equals(self.acc_mx.index) and self.towns.index.equals(self.acc_mx.columns)):
            n = len(self.towns)
            self.towns = self.towns.reset_index(drop=True)
            self.acc_mx.index = pd.RangeIndex(n)
            self.acc_mx.columns = pd.RangeIndex(n)

        if st_path.exists():
            self.service_types_df = pd.read_pickle(st_path)
        if norm_path.exists():
            self.normatives_df = pd.read_pickle(norm_path)

        self.provision = ProvisionModel(self.towns, self.acc_mx, verbose=False)

    def _init_subsystems(self) -> None:
        self._set_infrastructure2()
        self._set_districts()
        self._set_towns()
        self._set_services()

    def _set_infrastructure2(self) -> None:
        if SERVICE_GROUPS_PLACEHOLDER:
            self.INFRASTRUCTURE = {name: list(services) for name, services in SERVICE_GROUPS_PLACEHOLDER.items()}
            return
        if self.service_types_df is None or self.service_types_df.empty:
            self.INFRASTRUCTURE = {}
            return
        df = self.service_types_df
        cat_col = "category" if "category" in df.columns else (
            "infrastructure_type" if "infrastructure_type" in df.columns else None
        )
        if cat_col is None:
            self.INFRASTRUCTURE = {"������": sorted(df["name"].dropna().unique().tolist())}
            return
        infra_to_names = (
            df.groupby(cat_col)["name"].apply(lambda s: sorted(pd.Series(s.dropna().unique()).tolist())).to_dict()
        )
        mapping_ru = {
            "basic": "�������",
            "comfort": "�������",
            None: "������",
        }
        self.INFRASTRUCTURE = {mapping_ru.get(k, str(k)): v for k, v in infra_to_names.items()}

    def _set_infrastructure(self) -> None:
        """Backward-compatible alias for legacy callers (townsnet < 10)."""
        self._set_infrastructure2()

    def _set_services(self) -> None:
        # В новом API нет источника объектов услуг — оставим пустую заготовку
        if self.towns is not None:
            self.services = gpd.GeoDataFrame(
                {"town": [], "service_type": [], "geometry": [], "capacity": []},
                geometry="geometry",
                crs=self.towns.crs,
            )
        else:
            self.services = None

    def _set_districts(self) -> None:
        # Нет данных о районах — пропускаем
        self.municipal_districts = None

    def _set_towns(self) -> None:
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not loaded.")
        if "town_name" not in self.towns.columns and "name" in self.towns.columns:
            self.towns = self.towns.rename(columns={"name": "town_name"})

    # ----------------- Построение производных -----------------

    def _build_flows(self) -> None:
        """Построить гео-«стрелки» потоков из combined_matrix."""
        if self.combined_matrix is None or self.towns is None:
            raise RuntimeError("Matrix or towns are not initialized.")
        if not issparse(self.combined_matrix):
            raise TypeError("combined_matrix must be sparse CSR/CSC matrix.")

        rows, cols, values = find(self.combined_matrix)
        flows: List[Dict] = []
        for i, j, val in zip(rows, cols, values):
            if i not in self.towns.index or j not in self.towns.index:
                continue
            from_point = self.towns.loc[i, "geometry"].centroid
            to_point = self.towns.loc[j, "geometry"].centroid
            flows.append(
                {
                    "from_id": i,
                    "to_id": j,
                    "from_name": self.towns.loc[i, "town_name"],
                    "to_name": self.towns.loc[j, "town_name"],
                    "demand": float(val),
                    "geometry": LineString([from_point, to_point]),
                }
            )

        self.flows = gpd.GeoDataFrame(flows, geometry="geometry", crs=self.towns.crs)

    def update_population(self, population: pd.DataFrame) -> None:
        """
        Обновить население городов по таблице population[region_city, population].
        """
        if self.towns is None:
            raise RuntimeError("Model is not initialized.")
        # сопоставляем по имени
        mask = self.towns["town_name"].isin(population["region_city"].to_list())
        idxs = self.towns[mask].index.tolist()
        for i in idxs:
            city_name = self.towns.loc[i, "town_name"]
            pop = population.loc[population["region_city"] == city_name, "population"]
            if not pop.empty:
                self.towns.at[i, "population"] = pop.iloc[0]
        # перезагрузим towns с обновлённым населением
        self._set_towns()

    def calculate_ueqi(self) -> gpd.GeoDataFrame:
        """
        Рассчитать UEQI-группы по городам (на 100 жителей).
        """
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not initialized.")
        towns_copy = self.towns.copy()

        for group_name, services in UEQI_GROUPS.items():
            existing = [s for s in services if s in towns_copy.columns]
            if not existing:
                towns_copy[group_name] = 0.0
                continue
            towns_copy[group_name] = towns_copy[existing].sum(axis=1) / (towns_copy["population"] / 100)
            towns_copy[group_name] = np.minimum(100, towns_copy[group_name].round(3))

        return towns_copy
    
    def update_service(self, name: str, new_capacity: float, params: Dict[str, Sequence[str]]) -> str:
        """
        Добавить новый сервис (случайный тип из заданных групп) в городе с названием name.
        Смещение точки — случайное, контролируемое seed (если задан).
        Возвращает тип добавленного сервиса.
        """
        if self.towns is None or self.services is None:
            raise RuntimeError("Model is not initialized.")

        rng = default_rng(self.seed)

        # выбираем группу и тип
        if not params:
            raise ValueError("params must contain at least one service group.")
        service_key = rng.choice(list(params.keys()))
        # TODO: убрать хак [9:] как только введёшь нормальные ключи в UEQI_GROUPS
        type_of_service = rng.choice([s[9:] for s in UEQI_GROUPS[service_key]])

        town_row = self.towns[self.towns["town_name"].str.contains(name, case=False)]
        if town_row.empty:
            raise KeyError(f"Town with name '{name}' not found in towns data.")

        town_id = town_row.index.item()
        town_point = town_row.geometry.item()

        # существующие сервисы такого типа
        service_mask = (self.services["town"].str.contains(name, case=False)) & (
            self.services["service_type"] == type_of_service
        )
        service = self.services.loc[service_mask, ["geometry", "capacity"]].copy()
        service["town_id"] = town_id

        # случайное смещение вокруг центра города
        offset_x = rng.uniform(-OFFSET_PX, OFFSET_PX)
        offset_y = rng.uniform(-OFFSET_PX, OFFSET_PX)
        new_point = Point(town_point.x + offset_x, town_point.y + offset_y)

        new_service = gpd.GeoDataFrame(
            {"town_id": [town_id], "geometry": [new_point], "capacity": [new_capacity]},
            crs=self.towns.crs,
        )

        updated_services = pd.concat([service, new_service], ignore_index=True)

        # persist locally only
        self.services = gpd.GeoDataFrame(updated_services, geometry="geometry", crs=self.towns.crs)
        return type_of_service
    
    def load_migration_matrix(
        self,
        *,
        matrix_dir: str | Path = DEFAULT_MATRIX_DIR,
        average: bool = True,
        anchors: Optional[pd.DataFrame] = None,
    ) -> None:
        """Load aggregated provision outputs from parquet files into memory."""
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not initialized.")

        base_path = Path(matrix_dir)
        _ = average  # kept for backward compatibility
        candidates = list(self.INFRASTRUCTURE.keys()) if self.INFRASTRUCTURE else []
        self.group_provision = {}
        self.group_matrices = {}
        self.group_flows = {}

        for group_name in candidates:
            file_path = base_path / f"{group_name}_provision.parquet"
            if not file_path.exists():
                updated_path = base_path / UPDATED_SUBDIR / f"{group_name}_provision.parquet"
                file_path = updated_path if updated_path.exists() else file_path
            if not file_path.exists():
                continue
            df = pd.read_parquet(file_path)
            self.group_provision[group_name] = df

        self._ensure_anchor_frame(anchors)

    def calculate_provision(
        self,
        services: Optional[Sequence[str]] = None,
        data_path: str | Path | None = None,
        *,
        supplies_by_service: Optional[Dict[str | int, pd.DataFrame]] = None,
        services_source: str | Path | gpd.GeoDataFrame | None = None,
    ) -> None:
        """Run provision assessment for the requested service groups and store aggregated coverage."""
        if self.provision is None or self.towns is None:
            raise RuntimeError("Model is not initialized.")

        self._ensure_service_types()
        self._set_towns()

        infra = dict(self.INFRASTRUCTURE)
        if not infra and self.service_types_by_name:
            infra = {name: [name] for name in self.service_types_by_name.keys()}

        requested = (
            list(infra.keys())
            if services is None
            else ([services] if isinstance(services, str) else list(services))
        )

        target_groups: Dict[str, List[str]] = {}
        missing_names: List[str] = []
        name_lookup = {name.lower(): st for name, st in self.service_types_by_name.items()}

        for group_name in requested:
            if group_name in infra:
                target_groups[group_name] = list(infra[group_name])
            else:
                key = str(group_name)
                st = self.service_types_by_name.get(key) or name_lookup.get(key.lower())
                if st is not None:
                    target_groups[key] = [st.name]
                else:
                    try:
                        st_id = int(key)
                    except (TypeError, ValueError):
                        st_id = None
                    if st_id is not None and st_id in (self.service_types or {}):
                        st_obj = self.service_types[st_id]
                        target_groups[st_obj.name] = [st_obj.name]
                    else:
                        missing_names.append(key)

        if missing_names and not target_groups:
            raise KeyError(f"Unknown service groups or types: {', '.join(missing_names)}")
        if not target_groups:
            return

        out_dir = Path(data_path) if data_path is not None else DEFAULT_MATRIX_DIR / UPDATED_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)

        def _normalize_supply(df: pd.DataFrame) -> pd.DataFrame:
            if "supply" not in df.columns:
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: "supply"})
                else:
                    raise ValueError("Supply DataFrame must contain a 'supply' column.")
            normalized = df.reindex(self.towns.index).copy()
            normalized["supply"] = pd.to_numeric(normalized["supply"], errors="coerce").fillna(0.0)
            return normalized

        supply_by_name: Dict[str, pd.DataFrame] = {}
        supply_by_id: Dict[int, pd.DataFrame] = {}

        if supplies_by_service:
            for key, df in supplies_by_service.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                normalized = _normalize_supply(df)
                if isinstance(key, int):
                    supply_by_id[key] = normalized
                else:
                    try:
                        supply_by_id[int(str(key))] = normalized
                    except (TypeError, ValueError):
                        pass
                supply_by_name[str(key).strip().lower()] = normalized

        if services_source is not None:
            geo_supplies = self._load_supplies_from_geojson(services_source)
            for service_name, df in geo_supplies.items():
                normalized = _normalize_supply(df)
                supply_by_name[str(service_name).strip().lower()] = normalized
                st_obj = self.service_types_by_name.get(service_name) or name_lookup.get(str(service_name).lower())
                if st_obj is not None:
                    supply_by_id[st_obj.id] = normalized

        population_series = (
            pd.to_numeric(self.towns.get("population"), errors="coerce")
            if "population" in self.towns.columns
            else pd.Series(0, index=self.towns.index)
        ).fillna(0.0).astype(float)

        base_frame = pd.DataFrame(
            {
                "city_id": self.towns.index,
                "city_name": self.towns["town_name"].astype(str).values,
                "population": population_series.values,
            }
        ).set_index("city_id")

        self.service_results = {}
        self.group_provision = {}
        self.group_matrices = {}
        self.group_flows = {}
        self.mobility = None
        self.city_json = {}

        for group_name, service_names in target_groups.items():
            group_df = base_frame.copy()
            weighted_sum = None
            weight_total = 0.0

            for service_name in service_names:
                key = str(service_name)
                st = (
                    self.service_types_by_name.get(key)
                    or name_lookup.get(key.lower())
                )
                if st is None:
                    try:
                        st_id = int(key)
                    except (TypeError, ValueError):
                        st_id = None
                    st = (self.service_types or {}).get(st_id) if st_id is not None else None
                if st is None:
                    raise KeyError(f"Unknown service type '{service_name}'")

                supply = (
                    supply_by_name.get(st.name.lower())
                    or supply_by_name.get(key.lower())
                    or supply_by_id.get(st.id)
                    or supply_by_name.get(str(st.id))
                )
                if supply is None:
                    supply = self._empty_supply_dataframe()
                else:
                    supply = supply.copy()

                result_gdf = self.provision.calculate(supply, st)
                result_gdf = result_gdf.copy()
                result_gdf["city_id"] = result_gdf.index
                result_gdf["city_name"] = self.towns.loc[result_gdf.index, "town_name"].values
                self.service_results[st.name] = result_gdf

                result_df = result_gdf.set_index("city_id")
                share = result_df.get("provision", pd.Series(0, index=group_df.index)).reindex(group_df.index).fillna(0.0)
                share = share.clip(lower=0.0)
                group_df[f"{st.name}_provision"] = share
                group_df[f"{st.name}_served"] = share * group_df["population"]

                weight = st.weight if st.weight is not None and st.weight > 0 else 1.0
                weighted_component = share * weight
                weighted_sum = weighted_component if weighted_sum is None else weighted_sum.add(weighted_component, fill_value=0.0)
                weight_total += weight

            if weight_total > 0 and weighted_sum is not None:
                group_share = weighted_sum / weight_total
            else:
                group_share = pd.Series(0.0, index=group_df.index)

            group_df["group_provision"] = group_share.clip(lower=0.0, upper=1.0)
            group_df["group_provision_pct"] = group_df["group_provision"] * 100.0
            group_df["group_served_population"] = (group_df["group_provision"] * group_df["population"]).round(2)
            group_df["group_name"] = group_name

            export_df = group_df.reset_index()
            self.group_provision[group_name] = export_df
            export_df.to_parquet(out_dir / f"{group_name}_provision.parquet", index=False)

        self._ensure_anchor_frame()
    def _compute_self_sufficiency(self, movement_matrix_csr: csr_matrix) -> Tuple[pd.DataFrame, np.ndarray]:
        """Вычисляет самообеспеченность каждого города."""
        if self.anchor is None:
            raise RuntimeError("Anchor GeoDataFrame is not initialized.")

        total_outflow = np.array(movement_matrix_csr.sum(axis=1)).ravel()
        population = self.anchor["population"].to_numpy()
        self_sufficiency_pct = np.round(((population - total_outflow) / population) * 100.0, 5)

        city_type: List[str] = []
        movement_csc = movement_matrix_csr.tocsc()
        for idx in self.anchor.index:
            self_pct = self_sufficiency_pct[idx]
            if self_pct > 0:
                inflow = float(movement_csc[:, idx].sum())
                city_type.append("градообразующий" if inflow > 0 else "градообслуживающий")
            else:
                city_type.append("не самодостаточный")

        df = pd.DataFrame(
            {
                "city_id": self.anchor.index,
                "city_name": self.anchor["town_name"].values,
                "population": population,
                "outflow": np.round(total_outflow, 5),
                "self_sufficiency_pct": self_sufficiency_pct,
                "city_type": city_type,
            }
        )
        return df, self_sufficiency_pct

    def _compute_anchor_coverage(
        self, movement_matrix_csr: csr_matrix, anchor_ids: Sequence[int], non_anchor_ids: Sequence[int]
    ) -> pd.DataFrame:
        """Покрытие неопорных городов опорными."""
        if self.anchor is None:
            raise RuntimeError("Anchor GeoDataFrame is not initialized.")
        movement_csc = movement_matrix_csr.tocsc()

        rows: List[Dict] = []
        for city_id in non_anchor_ids:
            city_name = self.anchor.at[city_id, "town_name"]
            to_all = float(movement_csc[city_id].sum())
            if to_all <= 0:
                continue
            for anchor_id in anchor_ids:
                to_anchor = float(movement_csc[city_id, anchor_id])
                if to_anchor >= 1.0:
                    to_others = to_all - to_anchor
                    coverage_pct = round((to_anchor / (to_anchor + to_others)) * 100.0, 5) if (to_anchor + to_others) > 0 else 0.0
                    rows.append(
                        {
                            "city_id": city_id,
                            "city_name": city_name,
                            "anchor_id": anchor_id,
                            "anchor_name": self.anchor.at[anchor_id, "town_name"],
                            "to_anchor": round(to_anchor, 5),
                            "to_other_non_anchors": round(to_others, 5),
                            "coverage_pct": coverage_pct,
                        }
                    )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _compute_anchor_stats(
        self, coverage_df: pd.DataFrame, self_sufficiency_df: pd.DataFrame, anchor_ids: Sequence[int]
    ) -> pd.DataFrame:
        """Статистика по опорным пунктам."""
        if self.anchor is None:
            raise RuntimeError("Anchor GeoDataFrame is not initialized.")

        stats: List[Dict] = []
        for anchor_id in anchor_ids:
            anchor_name = self.anchor.at[anchor_id, "town_name"]
            data = coverage_df[coverage_df["anchor_id"] == anchor_id]
            if not data.empty:
                stats.append(
                    {
                        "anchor_id": anchor_id,
                        "anchor_name": anchor_name,
                        "mean_coverage": round(data["coverage_pct"].mean(), 5),
                        "median_coverage": round(data["coverage_pct"].median(), 5),
                        "min_coverage": round(data["coverage_pct"].min(), 5),
                        "max_coverage": round(data["coverage_pct"].max(), 5),
                        "num_covered_cities": int(len(data)),
                    }
                )
            else:
                stats.append(
                    {
                        "anchor_id": anchor_id,
                        "anchor_name": anchor_name,
                        "mean_coverage": 0.0,
                        "median_coverage": 0.0,
                        "min_coverage": 0.0,
                        "max_coverage": 0.0,
                        "num_covered_cities": 0,
                    }
                )

        df = pd.DataFrame(stats)

        # self_sufficiency_df: индекс по city_id -> self_sufficiency_pct
        suff_by_id = self_sufficiency_df.set_index("city_id")["self_sufficiency_pct"]
        # сравниваем по anchor_id (исправил баг: раньше мапилось по имени)
        df["is_weak_anchor"] = df["anchor_id"].map(lambda cid: suff_by_id.get(cid, 100.0) < 95.0)
        return df

    def _compute_potential_anchors(
        self,
        movement_matrix_csr: csr_matrix,
        self_sufficiency_pct: np.ndarray,
        non_anchor_ids: Sequence[int],
        threshold: float,
    ) -> pd.DataFrame:
        """Определение потенциальных опорных пунктов."""
        if self.anchor is None:
            raise RuntimeError("Anchor GeoDataFrame is not initialized.")
        movement_csc = movement_matrix_csr.tocsc()

        rows: List[Dict] = []
        for col_id in non_anchor_ids:
            incoming = movement_csc[:, col_id].toarray().ravel()
            total_incoming = float(incoming.sum())
            if total_incoming < 1.0:
                continue

            self_pct = float(self_sufficiency_pct[col_id])
            if self_pct < threshold:
                continue

            # источники
            num_sources = int((incoming > 0).sum())
            from_non_anchors = float(incoming[non_anchor_ids].sum())
            rows.append(
                {
                    "city_id": col_id,
                    "city_name": self.anchor.at[col_id, "town_name"],
                    "incoming_from_others": round(total_incoming, 5),
                    "from_non_anchors": round(from_non_anchors, 5),
                    "from_anchors": round(total_incoming - from_non_anchors, 5),
                    "num_sources": num_sources,
                    "self_sufficiency_pct": round(self_pct, 5),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def analyze_mobility(self, anchor_threshold: float = 75.0) -> Dict[str, pd.DataFrame]:
        """Summarise provision results per group and identify potential anchor cities."""
        if not self.group_provision:
            raise RuntimeError("Run calculate_provision() or load_migration_matrix() first.")

        self._ensure_anchor_frame()
        anchor_threshold = float(anchor_threshold)

        group_mobility: Dict[str, Dict[str, pd.DataFrame]] = {}
        coverage_rows: List[pd.DataFrame] = []
        potential_rows: List[pd.DataFrame] = []

        for group_name, df in self.group_provision.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            coverage = df.copy()
            if "city_id" not in coverage.columns:
                coverage = coverage.reset_index().rename(columns={"index": "city_id"})
            if "group_provision" not in coverage.columns:
                if "provision_share" in coverage.columns:
                    coverage["group_provision"] = coverage["provision_share"]
                else:
                    continue
            if "group_provision_pct" not in coverage.columns:
                coverage["group_provision_pct"] = coverage["group_provision"] * 100.0
            if "group_served_population" not in coverage.columns:
                coverage["group_served_population"] = coverage["group_provision"] * coverage.get("population", 0)

            tidy = coverage[[
                "city_id",
                "city_name",
                "population",
                "group_provision",
                "group_provision_pct",
                "group_served_population",
            ]].copy()
            tidy["group"] = group_name
            tidy = tidy.rename(columns={
                "group_provision": "provision_share",
                "group_provision_pct": "provision_pct",
                "group_served_population": "served_population",
            })
            tidy["provision_pct"] = pd.to_numeric(tidy["provision_pct"], errors="coerce").fillna(0.0)
            tidy["served_population"] = pd.to_numeric(tidy["served_population"], errors="coerce").fillna(0.0)

            potential = tidy[tidy["provision_pct"] >= anchor_threshold].copy()

            group_mobility[group_name] = {
                "coverage": tidy,
                "potential_anchors": potential,
            }

            if not tidy.empty:
                coverage_rows.append(tidy)
            if not potential.empty:
                potential_rows.append(potential)

        coverage_all = pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
        potential_all = pd.concat(potential_rows, ignore_index=True) if potential_rows else pd.DataFrame()

        self.group_mobility = group_mobility
        self.mobility = {
            "coverage": coverage_all,
            "potential_anchors": potential_all,
            "anchor_threshold": anchor_threshold,
        }

        self._build_city_json()
        return self.mobility
    def _build_city_json(self) -> None:
        """Assemble city_json using aggregated provision metrics."""
        if self.anchor is None:
            self._ensure_anchor_frame()
        if not self.group_mobility:
            raise RuntimeError("Run analyze_mobility() first.")

        coverage_by_group: Dict[str, pd.DataFrame] = {}
        potential_union: set[int] = set()

        for group_name, payload in self.group_mobility.items():
            coverage = payload.get("coverage")
            if isinstance(coverage, pd.DataFrame) and not coverage.empty:
                coverage_by_group[group_name] = coverage.set_index("city_id")
            potential = payload.get("potential_anchors")
            if isinstance(potential, pd.DataFrame) and not potential.empty:
                potential_union.update(potential["city_id"].tolist())

        profiles: Dict[int, Dict[str, object]] = {}
        for city_id in self.anchor.index:
            row = self.anchor.loc[city_id]
            name = str(row.get("town_name", city_id))
            pop_val = row.get("population", 0)
            pop_val = int(float(pop_val)) if not pd.isna(pop_val) else 0
            is_anchor_city = bool(row.get("is_anchor_settlement", False))
            is_potential = (not is_anchor_city) and (city_id in potential_union)

            service_info: Dict[str, Dict[str, float | int]] = {}
            inflow_info: Dict[str, Dict[str, float]] = {}
            top_group = None
            top_value = -1.0

            for group_name, coverage in coverage_by_group.items():
                if city_id not in coverage.index:
                    continue
                cov_row = coverage.loc[city_id]
                pct = float(cov_row.get("provision_pct", 0.0))
                served = float(cov_row.get("served_population", 0.0))
                service_info[group_name] = {
                    "доля, %": round(pct, 2),
                    "население": int(round(served)),
                }
                inflow_info[group_name] = {
                    "приток": round(served, 2),
                    "доля, %": round(pct, 2),
                }
                if pct > top_value:
                    top_value = pct
                    top_group = group_name

            profile = {
                "Название": name,
                "Опорный пункт": is_anchor_city,
                "Потенциальный опорный пункт": is_potential,
                "Население": pop_val,
                "Самообеспеченность, %": round(top_value, 2) if top_value >= 0 else None,
                "Градообслуживающие функции": service_info,
                "Градообразующие функции": inflow_info,
                "Наиважнейшая градообразующая функция": top_group,
            }
            profiles[city_id] = profile

        self.city_json = profiles
    def save_city_json(self, path: str | Path, by: str = "id") -> None:
        """Save per-city JSON to file.

        by='id' -> keys are city_id (as strings).
        by='name' -> keys are city names.
        """
        if not self.city_json:
            self._build_city_json()

        if by == "name":
            data = {v["name"]: v for v in self.city_json.values()}
        else:
            data = {str(k): v for k, v in self.city_json.items()}

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_map(self, polygons_gdf: Optional[gpd.GeoDataFrame] = None):
        if self.combined_matrix is None or self.anchor is None or self.mobility is None:
            raise RuntimeError("Run load_migration_matrix() and analyze_mobility() first.")
        return create_anchor_flow_map(
            self.combined_matrix,
            self.anchor,
            self.mobility,
            polygons_gdf=polygons_gdf,
            group_mobility=self.group_mobility,
            group_matrices=self.group_matrices,
        )

    def save_static_map_png(
        self,
        out_path: str,
        polygons_gdf: Optional[gpd.GeoDataFrame] = None,
        *,
        focus: str = "anchors",
        max_edges: int = 5000,
        dpi: int = 200,
    ) -> str:
        if self.combined_matrix is None or self.anchor is None or self.mobility is None:
            raise RuntimeError("Run load_migration_matrix() and analyze_mobility() first.")
        return save_static_anchor_flow_png(
            self.combined_matrix,
            self.anchor,
            self.mobility,
            out_path,
            polygons_gdf=polygons_gdf,
            focus=focus,
            max_edges=max_edges,
            dpi=dpi,
        )
