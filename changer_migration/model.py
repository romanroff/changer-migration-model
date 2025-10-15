import os
from dataclasses import dataclass, field
from functools import singledispatchmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import concurrent.futures
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import csr_matrix, find, issparse
from shapely.geometry import LineString, Point
from tqdm.auto import tqdm

from townsnet import Region, Provision
from changer_migration.ueqi import UEQI_GROUPS
from changer_migration.creating_map import create_anchor_flow_map, save_static_anchor_flow_png

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

    # --- Групповые результаты ---
    group_matrices: Dict[str, csr_matrix] = field(default_factory=dict, init=False)
    group_flows: Dict[str, gpd.GeoDataFrame] = field(default_factory=dict, init=False)
    group_mobility: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict, init=False)

    # --- JSON profiles by city ---
    city_json: Dict[int, Dict] = field(default_factory=dict, init=False)


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
        # Build custom grouped INFRASTRUCTURE with Russian labels
        # Source columns: 'infrastructure' (category), 'name' (service type id)
        infra_to_names = (
            df.groupby("infrastructure")["name"].unique().apply(list).to_dict()
        )

        def get_names(keys: list[str]) -> list[str]:
            out: list[str] = []
            for k in keys:
                out.extend(infra_to_names.get(k, []))
            # keep order but unique
            seen = set()
            unique_out = []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    unique_out.append(x)
            return unique_out

        self.INFRASTRUCTURE = {
            "Безопасность": get_names(["SAFENESS"]),
            "Здравоохранение": get_names(["HEALTHCARE"]),
            "Культура": get_names(["LEISURE","RECREATION","COMMERCE",]),
            "Образование": get_names(["EDUCATION"]),
            "Спорт": get_names(["SPORT"]),
            "Туризм": get_names(["CATERING"]),
            "Услуги": get_names(["SERVICE"]),
        }

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
        *,
        matrix_dir: str | Path = DEFAULT_MATRIX_DIR,
        average: bool = True,
        anchors: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Загружает и агрегирует матрицы связей по всем группам из self.INFRASTRUCTURE.
        Для каждой группы формирует:
        • CSR-матрицу (self.group_matrices[group_name])
        • GeoDataFrame потоков (self.group_flows[group_name])
        Также формирует GeoDataFrame self.anchor (города + флаг якоря).
        """
        if self.towns is None:
            raise RuntimeError("Towns GeoDataFrame is not initialized.")

        total_nodes = self.towns.shape[0]
        base_path = Path(matrix_dir)

        global_report = {}

        for group_name, services in self.INFRASTRUCTURE.items():
            report = {"loaded": [], "missing": [], "errors": []}
            loaded_count = 0
            combined = csr_matrix((total_nodes, total_nodes), dtype=np.float64)

            for service in tqdm(services, desc=f"Группа {group_name}"):
                try:
                    filename = f"{service}_links.parquet"
                    file_path = base_path / filename

                    if file_path.exists():
                        df_rel = pd.read_parquet(file_path)
                        source = "original"
                    else:
                        report["missing"].append(service)
                        continue

                    valid = df_rel[
                        (df_rel["from"] < total_nodes) & (df_rel["to"] < total_nodes)
                    ]
                    rows = valid["from"].to_numpy()
                    cols = valid["to"].to_numpy()
                    data = valid["demand"].astype(np.float64).to_numpy()

                    combined += csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
                    loaded_count += 1
                    report["loaded"].append(f"{service} ({source})")
                except Exception as e:
                    report["errors"].append(f"Ошибка {service}: {e}")

            # усреднение по количеству сервисов
            if average and loaded_count > 0:
                combined /= loaded_count

            # сохранить матрицу группы
            self.group_matrices[group_name] = combined

            # построить потоки для данной группы
            self.combined_matrix = combined
            self._build_flows()
            self.group_flows[group_name] = self.flows.copy()

            global_report[group_name] = report

        # --- Сводный отчёт ---
        print("Результат загрузки групп матриц:")
        for group, rep in global_report.items():
            print(f"\n[{group}]")
            for k, v in rep.items():
                print(f"  {k}: {len(v)}")

        # --- Формирование self.anchor ---
        cities = self.towns.reset_index(drop=True)[["town_name", "geometry"]].copy()
        cities["population"] = self.towns["population"].values

        if anchors is not None and "is_anchor_settlement" in anchors.columns:
            anchor_flag = anchors["is_anchor_settlement"].astype(bool).reset_index(drop=True)
            anchor_flag = anchor_flag.reindex(range(len(cities)), fill_value=False)
        else:
            anchor_flag = pd.Series(False, index=range(len(cities)), name="is_anchor_settlement")

        cities["is_anchor_settlement"] = anchor_flag.values
        anchor_gdf = gpd.GeoDataFrame(cities, geometry="geometry", crs=self.towns.crs)
        anchor_gdf = anchor_gdf.reset_index(drop=True)
        anchor_gdf["city_id"] = anchor_gdf.index
        self.anchor = anchor_gdf.set_index("city_id")

    def calculate_provision(
        self,
        services: str | Sequence[str],
        data_path: str | Path | None = None,
        *,
        n_jobs: int = -1,
    ) -> None:
        """
        Пересчитать provision по заданным сервисам и сохранить parquet в data_path/updated.
        Можно распараллелить по сервисам, указав n_jobs. Значение -1 использует все доступные ядра.
        """
        if self.model is None or self.provision is None:
            raise RuntimeError("Model is not initialized.")
        services = [services] if isinstance(services, str) else list(services)

        out_dir = Path(data_path) if data_path is not None else DEFAULT_MATRIX_DIR / UPDATED_SUBDIR
        os.makedirs(out_dir, exist_ok=True)

        targets = [st for st in self.model.service_types if st.name in services]

        def _compute_and_save(st) -> str:
            # Создаём отдельный Provision на поток, чтобы избежать гонок состояния
            prov = Provision(region=self.model)
            _, _, _, l_gdf = prov.calculate(st)
            l_gdf.to_parquet(out_dir / f"{st.name}_links.parquet")
            return st.name

        if not targets:
            return

        if n_jobs == 1:
            for st in targets:
                _compute_and_save(st)
        else:
            workers = (os.cpu_count() or 1) if n_jobs == -1 else max(1, int(n_jobs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                fut2name = {ex.submit(_compute_and_save, st): st.name for st in targets}
                for fut in concurrent.futures.as_completed(fut2name):
                    name = fut2name[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[calculate_provision] Error while processing {name}: {e}")

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
        Анализ мобильности для каждой группы из `group_matrices` и для текущей `combined_matrix`.
        Для каждой матрицы вычисляются:
        - self_sufficiency
        - anchor_coverage
        - anchor_stats
        - potential_anchors

        Результаты по всем группам сохраняются в `self.group_mobility[group_name]`.
        Возвращает результаты для текущей `self.combined_matrix` (для совместимости с существующим кодом).
        """
        if self.anchor is None:
            raise RuntimeError("anchor GeoDataFrame is not initialized.")

        def _analyze_for_matrix(matrix: csr_matrix) -> Dict[str, pd.DataFrame]:
            movement_matrix_csr = csr_matrix(matrix)

            mm = movement_matrix_csr.tolil()
            mm.setdiag(0)
            movement_matrix_csr = mm.tocsr()

            anchor_ids = self.anchor[self.anchor["is_anchor_settlement"]].index.tolist()
            non_anchor_ids = self.anchor[~self.anchor["is_anchor_settlement"]].index.tolist()

            result: Dict[str, pd.DataFrame] = {}

            self_sufficiency_df, self_sufficiency_pct = self._compute_self_sufficiency(movement_matrix_csr)
            result["self_sufficiency"] = self_sufficiency_df

            coverage_df = self._compute_anchor_coverage(movement_matrix_csr, anchor_ids, non_anchor_ids)
            result["anchor_coverage"] = coverage_df

            anchor_stats_df = self._compute_anchor_stats(coverage_df, self_sufficiency_df, anchor_ids)
            result["anchor_stats"] = anchor_stats_df

            potential_anchors_df = self._compute_potential_anchors(
                movement_matrix_csr, self_sufficiency_pct, non_anchor_ids, anchor_threshold
            )
            result["potential_anchors"] = potential_anchors_df

            return result

        # 1) Посчитать по всем группам (если они есть)
        self.group_mobility = {}
        if self.group_matrices:
            for group_name, matrix in self.group_matrices.items():
                try:
                    self.group_mobility[group_name] = _analyze_for_matrix(matrix)
                except Exception as e:
                    # Не прерываем весь анализ, но сообщаем в консоль
                    print(f"[analyze_mobility] Ошибка в группе '{group_name}': {e}")

        # 2) Совместимость: вернуть и сохранить результаты для текущей combined_matrix
        if self.combined_matrix is None:
            # Если combined_matrix не задана, но группы посчитаны —
            # вернём последние результаты группы (если есть)
            if self.group_mobility:
                last_group = next(reversed(self.group_mobility))
                self.mobility = self.group_mobility[last_group]
                return self.mobility
            raise RuntimeError("combined_matrix is not initialized.")

        mobility_current = _analyze_for_matrix(self.combined_matrix)
        self.mobility = mobility_current

        # Build/update per-city JSON profile after analysis
        try:
            self._build_city_json()
        except Exception as e:
            # Non-fatal; still return analysis results
            print(f"[analyze_mobility] Failed to build city_json: {e}")
        return self.group_mobility

    def _build_city_json(self) -> None:
        """Builds a JSON-like dict with full city info for map/tooltips.

        Produces self.city_json keyed by city_id with structure:
        {
          city_id: {
            'name': str,
            'опорный пункт': bool,
            'потенциальный опорный пункт': bool,
            'Население': int,
            'Самообеспеченность, %': float,
            'Градообслуживающие функции': { group: {'доля, %': float, 'население': int} },
            'Градообразующие функции': { group: {'приток': float, 'доля, %': float} },
            'Наиважнейшая градообразующая функция': str | None,
          }
        }
        """
        if self.anchor is None or self.mobility is None:
            raise RuntimeError("Run load_migration_matrix() and analyze_mobility() first.")

        n = len(self.anchor)
        # Potential anchors (by combined matrix)
        pot_df = self.mobility.get("potential_anchors", pd.DataFrame())
        potential_ids = set(pot_df["city_id"].tolist()) if (isinstance(pot_df, pd.DataFrame) and not pot_df.empty) else set()

        # Overall self-sufficiency by city
        ss_df = self.mobility.get("self_sufficiency", pd.DataFrame())
        if not isinstance(ss_df, pd.DataFrame) or ss_df.empty:
            raise RuntimeError("Self sufficiency data is missing after analyze_mobility().")
        ss_df = ss_df.set_index("city_id")

        # Per-group self sufficiency shares (0..1)
        per_group_self: Dict[str, pd.Series] = {}
        for grp, res in self.group_mobility.items():
            df_self = res.get("self_sufficiency", pd.DataFrame())
            if df_self is None or df_self.empty:
                continue
            s = df_self.set_index("city_id")["self_sufficiency_pct"] / 100.0
            per_group_self[grp] = s

        # Per-group inflow arrays
        per_group_inflow: Dict[str, np.ndarray] = {}
        for grp, mat in self.group_matrices.items():
            try:
                csc = csr_matrix(mat).tocsc()
                inflow = np.asarray(csc.sum(axis=0)).ravel()
                per_group_inflow[grp] = inflow
            except Exception:
                continue

        profiles: Dict[int, Dict] = {}
        # Iterate in stable order of anchor index
        for city_id in self.anchor.index:
            row = self.anchor.loc[city_id]
            name = str(row["town_name"]) if "town_name" in row else str(city_id)
            pop_val = int(row.get("population", 0)) if not pd.isna(row.get("population", np.nan)) else 0
            is_anchor_city = bool(row.get("is_anchor_settlement", False))
            is_potential = (not is_anchor_city) and (city_id in potential_ids)

            # Services (self-sufficiency per group)
            service_info: Dict[str, Dict[str, float | int]] = {}
            for grp in self.INFRASTRUCTURE.keys():
                share = None
                s = per_group_self.get(grp)
                if s is not None and city_id in s.index:
                    share = float(s.loc[city_id])
                if share is None or np.isnan(share):
                    continue
                served = int(round(pop_val * share)) if pop_val else 0
                service_info[grp] = {
                    "доля, %": round(share * 100.0, 2),
                    "население": served,
                }

            # Industry (inflow per group + shares)
            inflow_info: Dict[str, Dict[str, float]] = {}
            total_inflow = 0.0
            for grp in self.INFRASTRUCTURE.keys():
                infl = per_group_inflow.get(grp)
                val = float(infl[city_id]) if infl is not None and city_id < len(infl) else 0.0
                total_inflow += val
            top_group = None
            top_value = -1.0
            for grp in self.INFRASTRUCTURE.keys():
                infl = per_group_inflow.get(grp)
                val = float(infl[city_id]) if infl is not None and city_id < len(infl) else 0.0
                pct = (val / total_inflow * 100.0) if total_inflow > 0 else 0.0
                inflow_info[grp] = {"приток": round(val, 2), "доля, %": round(pct, 2)}
                if val > top_value:
                    top_value = val
                    top_group = grp

            profile = {
                "Название": name,
                "Опорный пункт": is_anchor_city,
                "Потенциальный опорный пункт": is_potential,
                "Население": pop_val,
                "Самообеспеченность, %": float(ss_df.at[city_id, "self_sufficiency_pct"]) if city_id in ss_df.index else None,
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


