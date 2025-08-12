import os
import itertools
from dataclasses import dataclass, field
from functools import singledispatchmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import csr_matrix, find, issparse
from shapely.geometry import LineString, Point
from tqdm.auto import tqdm

from townsnet import Region, Provision
from changer_migration.ueqi import UEQI_GROUPS
from changer_migration.creating_map import create_anchor_flow_map

# ---- Константы/настройки ----
OFFSET_PX: float = 300.0  # сдвиг при генерации новой точки сервиса на N метров
DEFAULT_MATRIX_DIR = Path("data/provision")  # базовый путь для матриц
UPDATED_SUBDIR = "updated"  # подкаталог с обновлёнными матрицами

@dataclass
class MigrationFlowModel:
    """
    Модель потоков миграции и анализа «якорных» городов.
    """

    # --- Внешние объекты-модели ---
    model: Optional[Region] = field(default=None, init=False)
    provision: Optional[Provision] = field(default=None, init=False)

    # --- Основные таблицы/геоданные ---
    towns: Optional[gpd.GeoDataFrame] = field(default=None, init=False)
    services: Optional[gpd.GeoDataFrame] = field(default=None, init=False)
    municipal_districts: Optional[gpd.GeoDataFrame] = field(default=None, init=False)

    # --- Справочники/инфраструктура ---
    INFRASTRUCTURE: Dict[str, List[str]] = field(default_factory=dict, init=False)

    # --- Матрицы/производные данные ---
    combined_matrix: Optional[csr_matrix] = field(default=None, init=False)
    flows: Optional[gpd.GeoDataFrame] = field(default=None, init=False)
    anchor: Optional[gpd.GeoDataFrame] = field(default=None, init=False)  # города + признаки якоря
    mobility: Optional[Dict[str, pd.DataFrame]] = field(default=None, init=False)

    # --- Параметры/seed ---
    seed: Optional[int] = None  # детерминизм по желанию


    # ----------------- Конструирование/загрузка -----------------

    @classmethod
    def from_pickle(cls, path: str | Path, *, seed: Optional[int] = None) -> "MigrationFlowModel":
        """
        Фабричный метод: загрузка Region из pickle и подготовка зависимостей.
        """
        m = cls(seed=seed)
        m._load_region(path)
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

    def _load_region(self, path: str | Path) -> None:
        self.model = Region.from_pickle(path)
        self.provision = Provision(region=self.model)

    def _init_subsystems(self) -> None:
        self._set_infrastructure()
        self._set_districts()
        self._set_towns()
        self._set_services()

    def _set_infrastructure(self) -> None:
        assert self.model is not None
        df = self.model.get_service_types_df()
        self.INFRASTRUCTURE = df.groupby("infrastructure")["name"].unique().apply(list).to_dict()

    def _set_services(self) -> None:
        assert self.model is not None
        self.services = self.model.get_services_gdf()

    def _set_districts(self) -> None:
        assert self.model is not None
        self.municipal_districts = self.model.districts

    def _set_towns(self) -> None:
        assert self.model is not None
        self.towns = self.model.get_towns_gdf()

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
        if self.towns is None or self.model is None:
            raise RuntimeError("Model is not initialized.")
        # сопоставляем по имени
        mask = self.towns["town_name"].isin(population["region_city"].to_list())
        idxs = self.towns[mask].index.tolist()
        for i in idxs:
            city_name = self.model[i].name
            pop = population.loc[population["region_city"] == city_name, "population"]
            if not pop.empty:
                self.model[i].population = pop.iloc[0]
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
        if self.towns is None or self.services is None or self.model is None:
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
            crs=self.model.crs,
        )

        updated_services = pd.concat([service, new_service], ignore_index=True)

        self.model.update_services(type_of_service, updated_services)
        self._set_services()
        return type_of_service
    
    def load_migration_matrix(
        self,
        infra_keys: str | Sequence[str],
        *,
        matrix_dir: str | Path = DEFAULT_MATRIX_DIR,
        use_updated: bool = False,
        average: bool = True,
        anchors: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Загрузить и агрегировать матрицы связей по нужным сервисам в CSR-матрицу.
        Также формирует GeoDataFrame self.anchor (города + флаг якоря).
        """
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not initialized.")

        total_nodes = self.towns.shape[0]
        keys_flat = [infra_keys] if isinstance(infra_keys, str) else list(
            itertools.chain.from_iterable([k] if isinstance(k, str) else k for k in infra_keys)
        )

        # валидация ключей
        invalid = [k for k in keys_flat if k not in self.INFRASTRUCTURE]
        if invalid:
            raise ValueError(f"Неверные ключи: {invalid}. Доступные: {list(self.INFRASTRUCTURE.keys())}")

        all_services = list(itertools.chain.from_iterable(self.INFRASTRUCTURE[k] for k in keys_flat))

        report = {"loaded": [], "missing": [], "errors": []}
        loaded_count = 0

        self.combined_matrix = csr_matrix((total_nodes, total_nodes), dtype=np.float64)

        base_path = Path(matrix_dir)
        for key in tqdm(all_services, desc="Загрузка матриц связей"):
            try:
                filename = f"{key}_links.parquet"
                updated_path = base_path / UPDATED_SUBDIR / filename
                file_path = base_path / filename

                if use_updated and updated_path.exists():
                    df_rel = pd.read_parquet(updated_path)
                    source = "updated"
                elif file_path.exists():
                    df_rel = pd.read_parquet(file_path)
                    source = "original"
                else:
                    report["missing"].append(key)
                    continue

                valid = df_rel[(df_rel["from"] < total_nodes) & (df_rel["to"] < total_nodes)]
                rows = valid["from"].to_numpy()
                cols = valid["to"].to_numpy()
                data = valid["demand"].astype(np.float64).to_numpy()

                self.combined_matrix += csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
                loaded_count += 1
                report["loaded"].append(f"{key} ({source})")
            except Exception as e:
                report["errors"].append(f"Ошибка загрузки {key}: {e}")

        if average and loaded_count > 0:
            self.combined_matrix /= loaded_count
        print(f"Результат загрузки матриц связей:\n{report}")
        
        # построим GeoDataFrame self.flows
        self._build_flows()

        # соберём self.anchor (города + флаг якорности)
        cities = self.towns.reset_index(drop=True)[["town_name", "geometry"]].copy()
        cities["population"] = self.towns["population"].values

        if anchors is not None and "is_anchor_settlement" in anchors.columns:
            # приведём индексы в общий вид
            anchor_flag = anchors["is_anchor_settlement"].astype(bool).reset_index(drop=True)
            anchor_flag = anchor_flag.reindex(range(len(cities)), fill_value=False)
        else:
            anchor_flag = pd.Series(False, index=range(len(cities)), name="is_anchor_settlement")

        cities["is_anchor_settlement"] = anchor_flag.values
        anchor_gdf = gpd.GeoDataFrame(cities, geometry="geometry", crs=self.towns.crs)
        anchor_gdf = anchor_gdf.reset_index(drop=True)
        anchor_gdf["city_id"] = anchor_gdf.index
        self.anchor = anchor_gdf.set_index("city_id")

    def calculate_provision(self, services: str | Sequence[str], data_path: str | Path | None = None) -> None:
        """
        Пересчитать provision по заданным сервисам и сохранить parquet в data_path/updated.
        """
        if self.model is None or self.provision is None:
            raise RuntimeError("Model is not initialized.")
        services = [services] if isinstance(services, str) else list(services)

        out_dir = Path(data_path) if data_path is not None else DEFAULT_MATRIX_DIR / UPDATED_SUBDIR
        os.makedirs(out_dir, exist_ok=True)

        for service_type in self.model.service_types:
            if service_type.name in services:
                _, _, _, l_gdf = self.provision.calculate(service_type)
                print(f"✔ {service_type.name:<15} was processed")
                l_gdf.to_parquet(out_dir / f"{service_type.name}_links.parquet")

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
        """
        Анализ мобильности: самообеспеченность, покрытие опорными пунктами, статистика и потенциальные опорные.
        Возвращает словарь с DataFrame.
        """
        if self.combined_matrix is None:
            raise RuntimeError("combined_matrix is not initialized.")
        if self.anchor is None:
            raise RuntimeError("anchor GeoDataFrame is not initialized.")

        movement_matrix_csr = csr_matrix(self.combined_matrix)

        # обнулим диагональ
        mm = movement_matrix_csr.tolil()
        mm.setdiag(0)
        movement_matrix_csr = mm.tocsr()

        anchor_ids = self.anchor[self.anchor["is_anchor_settlement"]].index.tolist()
        non_anchor_ids = self.anchor[~self.anchor["is_anchor_settlement"]].index.tolist()

        mobility: Dict[str, pd.DataFrame] = {}

        self_sufficiency_df, self_sufficiency_pct = self._compute_self_sufficiency(movement_matrix_csr)
        mobility["self_sufficiency"] = self_sufficiency_df

        coverage_df = self._compute_anchor_coverage(movement_matrix_csr, anchor_ids, non_anchor_ids)
        mobility["anchor_coverage"] = coverage_df

        anchor_stats_df = self._compute_anchor_stats(coverage_df, self_sufficiency_df, anchor_ids)
        mobility["anchor_stats"] = anchor_stats_df

        potential_anchors_df = self._compute_potential_anchors(
            movement_matrix_csr, self_sufficiency_pct, non_anchor_ids, anchor_threshold
        )
        mobility["potential_anchors"] = potential_anchors_df

        self.mobility = mobility
        return mobility

    def create_map(self, polygons_gdf: Optional[gpd.GeoDataFrame] = None):
        if self.combined_matrix is None or self.anchor is None or self.mobility is None:
            raise RuntimeError("Run load_migration_matrix() and analyze_mobility() first.")
        return create_anchor_flow_map(self.combined_matrix, self.anchor, self.mobility, polygons_gdf=polygons_gdf)


