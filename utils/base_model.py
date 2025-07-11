import os
from typing import Optional
from scipy.sparse import csr_matrix, find
from tqdm.auto import tqdm
from pathlib import Path
from functools import singledispatchmethod
import itertools
import random
from shapely import LineString, Point
from townsnet import Region, Provision
from utils.ueqi import UEQI_GROUPS
from utils.creating_map import create_anchor_flow_map
import geopandas as gpd
import pandas as pd
import numpy as np

class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, town_id):
        if not town_id in self.towns.index:
            raise KeyError(f"Can't find town with such id: {town_id}")
        return self.towns.loc[town_id]

    def from_pickle(self, path: str):
        self.model = Region.from_pickle(path)
        self.provision = Provision(region=self.model)
        self._get_infrastructure()
        self._get_districts()
        self._get_towns()
        self._get_services()

    def _get_infrastructure(self):
        self.INFRASTRUCTURE = self.model.get_service_types_df().groupby('infrastructure')['name'].unique().to_dict()

    def _get_services(self):
        self.services = self.model.get_services_gdf()

    def _get_districts(self):
        self.municipal_districts = self.model.districts
    
    def _get_towns(self):
        self.towns = self.model.get_towns_gdf()
    
    def _get_flows(self):
        rows, cols, values = find(self.combined_matrix)

        # Теперь создадим DataFrame с from_id, to_id и соответствующими точками
        flows = []
        for i, j, val in zip(rows, cols, values):
            try:
                from_point = self.towns.loc[i, 'geometry'].centroid
                to_point = self.towns.loc[j, 'geometry'].centroid
            except KeyError:
                continue
            
            flows.append({
                "from_id": i,
                "to_id": j,
                "from_name": self.towns.loc[i, 'town_name'],
                "to_name": self.towns.loc[j, 'town_name'],
                "demand": val,
                "geometry": LineString([from_point, to_point])
            })

        # Создаем GeoDataFrame с потоками между городами
        self.flows = gpd.GeoDataFrame(flows, geometry='geometry', crs=self.towns.crs)

    def update_population(self, population: pd.DataFrame):
        indexes = self.towns[self.towns.town_name.isin(population.region_city.to_list())].index.tolist()
        for i in indexes:
            self.model[i].population = population[population['region_city'] == self.model[i].name].population.iloc[0]
        self._get_towns()

    def calculate_ueqi(self):
        towns_copy = self.towns.copy()

        for group_name, services in UEQI_GROUPS.items():
            existing_services = [s for s in services if s in towns_copy.columns]
            towns_copy[group_name] = towns_copy[existing_services].sum(axis=1) / (towns_copy['population'] / 100)
            towns_copy[group_name] = np.minimum(100, round(towns_copy[group_name], 3))

        return towns_copy
    
    def update_service(self, name, new_capacity, params):
        """
        Добавляет случайный сервис из подходящей группы в случайном месте города.
        
        :param name: Название города
        :param new_capacity: Пропускная способность нового сервиса
        :param params: Словарь параметров для выбора типа сервиса
        :return: Тип добавленного сервиса
        """
        # Выбираем случайную группу и тип сервиса
        service_key = random.choice(list(params.keys()))
        type_of_service = random.choice(UEQI_GROUPS[service_key])[9:] # TODO: заменить [9:] на что-то адекватное

        # Фильтруем данные через self
        town_mask = self.towns['town_name'].str.contains(name, case=False)
        town_row = self.towns[town_mask]

        if town_row.empty:
            raise KeyError(f"Town with name '{name}' not found in towns data.")

        town_id = town_row.index.item()
        town_geometry = town_row.geometry.item()

        # Берём существующие сервисы этого типа в этом городе
        service_mask = (self.services.town.str.contains(name)) & (self.services['service_type'] == type_of_service)
        service = self.services[service_mask][['geometry', 'capacity']].copy()
        service['town_id'] = town_id

        # Получаем координаты центроида города
        center_x = town_geometry.x
        center_y = town_geometry.y

        # Генерируем случайное смещение
        offset_x = random.uniform(-300, 300)
        offset_y = random.uniform(-300, 300)

        # Создаём новую точку
        new_service_point = Point(center_x + offset_x, center_y + offset_y)

        # Создаём GeoDataFrame для нового сервиса
        new_service = gpd.GeoDataFrame(
            {
                'town_id': [town_id],
                'geometry': [new_service_point],
                'capacity': [new_capacity],
            },
            crs=self.model.crs  # Предполагается, что у BaseModel есть атрибут crs
        )

        # Объединяем старые и новые сервисы
        updated_services = pd.concat([service, new_service], ignore_index=True)

        # Обновляем данные в модели
        self.model.update_services(type_of_service, updated_services)
        self._get_services()

        return type_of_service
    
    def load_migration_matrix(self, infra_keys, matrix_dir="data/provision", use_updated=False, average=True, anchors=None):

        total_nodes = self.towns.shape[0]
        if isinstance(infra_keys, str):
            keys_flat = [infra_keys]
        else:
            keys_flat = list(itertools.chain.from_iterable(
                [k] if isinstance(k, str) else k for k in infra_keys
            ))

        # Проверка ключей
        invalid_keys = [k for k in keys_flat if k not in self.INFRASTRUCTURE]
        if invalid_keys:
            raise ValueError(
                f"Неверные ключи: {invalid_keys}. Доступные: {list(self.INFRASTRUCTURE.keys())}"
            )

        all_services = list(itertools.chain.from_iterable(
            self.INFRASTRUCTURE[k] for k in keys_flat
        ))

        report = {
            'loaded': [],
            'missing': [],
            'errors': []
        }
        loaded_count = 0

        # Создаем шаблон разреженной матрицы
        self.combined_matrix = csr_matrix((total_nodes, total_nodes), dtype=np.float64)

        for key in tqdm(all_services, desc=f"Загрузка матриц связей"):
            try:
                base_path = Path(matrix_dir)
                filename = f"{key}_links.parquet"

                file_path = base_path / filename
                updated_path = base_path / "updated" / filename

                # Сначала пробуем загрузить из updated, если флаг включён и файл существует
                if use_updated and updated_path.exists():
                    df_relations = pd.read_parquet(updated_path)
                    source = "updated"
                elif file_path.exists():
                    df_relations = pd.read_parquet(file_path)
                    source = "original"
                else:
                    report['missing'].append(key)
                    continue

                # Фильтруем по границам
                valid_rows = df_relations[
                    (df_relations['from'] < total_nodes) &
                    (df_relations['to'] < total_nodes)
                ]

                rows = valid_rows['from'].values
                cols = valid_rows['to'].values
                data = valid_rows['demand'].astype(np.float64).values

                current_matrix = csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
                self.combined_matrix += current_matrix
                loaded_count += 1
                report['loaded'].append(f"{key} ({source})")

            except Exception as e:
                report['errors'].append(f"Ошибка загрузки {key}: {str(e)}")

        if average and loaded_count > 0:
            self.combined_matrix /= loaded_count
        print(f"Результат загрузки матриц связей:\n{report}")

        self._get_flows()

        # Создаем GeoDataFrame с дополнительной информацией
        index_df = pd.DataFrame(index=range(self.combined_matrix.shape[0]))
        if anchors is not None:
            cities_reset = pd.concat(
                [self.towns.reset_index(drop=True)[['town_name', 'geometry']], anchors['is_anchor_settlement']],
                axis=1
            )
            # Добавляем флаг только если он не существует в anchors
            if 'is_anchor_settlement' not in anchors.columns:
                cities_reset['is_anchor_settlement'] = False
        else:
            cities_reset = self.towns.reset_index(drop=True)[['town_name', 'geometry']]
            cities_reset['is_anchor_settlement'] = False

        matrix_with_cities = index_df.join(cities_reset, how='left')
        matrix_with_cities['population'] = self.towns['population'].values
        self.anchor = gpd.GeoDataFrame(matrix_with_cities, geometry='geometry', crs=self.towns.crs)
        if 'city_id' not in self.anchor.columns:
            self.anchor = self.anchor.reset_index(drop=True)
            self.anchor['city_id'] = self.anchor.index
            self.anchor = self.anchor.set_index('city_id')
    
    def calculate_provision(self, services: str | list, data_path=None):
        if type(services) is str:
            services = [services]

        data_path = f'data/provision/updated' if data_path is None else data_path

        # Создаем директорию, если её нет
        os.makedirs(data_path, exist_ok=True)

        # Обновляем service_types в регионе
        for service_type in self.model.service_types:
            if service_type.name in services:
                st_name = service_type.name
                # Вычисляем provision для текущего service_type
                _, _, _, l_gdf = self.provision.calculate(service_type)

                print(f"✔ {st_name:<15} was processed")
                l_gdf.to_parquet(os.path.join(data_path, f'{st_name}_links.parquet'))

    def _compute_self_sufficiency(self, movement_matrix_csr):
        """Вычисляет самообеспеченность каждого города."""
        total_outflow = np.array(movement_matrix_csr.sum(axis=1)).flatten()
        population = self.anchor['population'].values
        self_sufficiency_pct = ((population - total_outflow) / population * 100).round(5)

        city_type = []
        for idx in self.anchor.index:
            self_pct = self_sufficiency_pct[idx]
            if self_pct > 0:
                inflow = movement_matrix_csr[:, idx].sum()
                if inflow > 0:
                    city_type.append("градообразующий")
                else:
                    city_type.append("градообслуживающий")
            else:
                city_type.append("не самодостаточный")

        df = pd.DataFrame({
            'city_id': self.anchor.index,
            'city_name': self.anchor['town_name'].values,
            'population': self.anchor['population'].values,
            'outflow': total_outflow.round(5),
            'self_sufficiency_pct': self_sufficiency_pct,
            'city_type': city_type
        })

        return df, self_sufficiency_pct

    def _compute_anchor_coverage(self, movement_matrix_csr, anchor_ids, non_anchor_ids):
        """Покрытие неопорных городов опорными."""
        coverage_data = []

        movement_csc = movement_matrix_csr.tocsc()  # Для эффективного доступа по столбцам

        for city_id in non_anchor_ids:
            city_name = self.anchor.at[city_id, 'town_name']
            for anchor_id in anchor_ids:
                to_anchor = round(float(movement_csc[city_id, anchor_id]), 5)
                if to_anchor >= 1:
                    to_others = round(float(movement_csc[city_id].sum() - to_anchor), 5)
                    coverage_pct = round((to_anchor / (to_anchor + to_others)) * 100, 5) if (to_anchor + to_others) > 0 else 0
                    coverage_data.append({
                        'city_id': city_id,
                        'city_name': city_name,
                        'anchor_id': anchor_id,
                        'anchor_name': self.anchor.at[anchor_id, 'town_name'],
                        'to_anchor': to_anchor,
                        'to_other_non_anchors': to_others,
                        'coverage_pct': coverage_pct
                    })

        return pd.DataFrame(coverage_data) if coverage_data else pd.DataFrame()

    def _compute_anchor_stats(self, coverage_df, self_sufficiency_df, anchor_ids):
        """Статистика по опорным пунктам."""
        stats = []
        for anchor_id in anchor_ids:
            anchor_name = self.anchor.at[anchor_id, 'town_name']
            anchor_data = coverage_df[coverage_df['anchor_name'] == anchor_name]
            if not anchor_data.empty:
                stats.append({
                    'anchor_id': anchor_id,
                    'anchor_name': anchor_name,
                    'mean_coverage': round(anchor_data['coverage_pct'].mean(), 5),
                    'median_coverage': round(anchor_data['coverage_pct'].median(), 5),
                    'min_coverage': round(anchor_data['coverage_pct'].min(), 5),
                    'max_coverage': round(anchor_data['coverage_pct'].max(), 5),
                    'num_covered_cities': len(anchor_data)
                })
            else:
                stats.append({
                    'anchor_id': anchor_id,
                    'anchor_name': anchor_name,
                    'mean_coverage': 0.0,
                    'median_coverage': 0.0,
                    'min_coverage': 0.0,
                    'max_coverage': 0.0,
                    'num_covered_cities': 0
                })

        df = pd.DataFrame(stats)

        # anchor_self_suff = self_sufficiency_df[
        #     self_sufficiency_df['city_name'].isin(df['anchor_name'])
        # ].set_index('city_name')['self_sufficiency_pct']
        anchor_self_suff = self_sufficiency_df.set_index('city_id')['self_sufficiency_pct']

        df['is_weak_anchor'] = df['anchor_name'].map(lambda x: anchor_self_suff.get(x, 100) < 95.0)

        return df

    def _compute_potential_anchors(self, movement_matrix_csr, self_sufficiency_pct, non_anchor_ids, threshold):
        """Определение потенциальных опорных пунктов."""
        movement_csc = movement_matrix_csr.tocsc()
        potential = []

        for col_id in non_anchor_ids:
            incoming = movement_csc[:, col_id].toarray().flatten()
            total_incoming = round(incoming.sum(), 5)
            from_non_anchors = incoming[non_anchor_ids].sum()
            self_pct = self_sufficiency_pct[col_id]

            if total_incoming >= 1 and self_pct >= threshold:
                num_sources = int((incoming > 0).sum())

                potential.append({
                    'city_id': col_id,
                    'city_name': self.anchor.at[col_id, 'town_name'],
                    'incoming_from_others': total_incoming,
                    'from_non_anchors': round(from_non_anchors, 5),
                    'from_anchors': round(total_incoming - from_non_anchors, 5),
                    'num_sources': num_sources,
                    'self_sufficiency_pct': round(self_pct, 5)
                })

        return pd.DataFrame(potential) if potential else pd.DataFrame()

    def analyze_mobility(self, anchor_threshold=75):
        """
        Анализ мобильности: самообеспеченность, покрытие опорными пунктами, статистика и потенциальные опорные.

        Parameters:
            gdf (gpd.GeoDataFrame): Информация о городах ['name', 'is_anchor_settlement', 'geometry', 'population']
            movement_matrix_csr (csr_matrix): Разреженная матрица перемещений (from -> to)
            anchor_threshold (float): Порог для потенциальных опорных пунктов (%)

        Returns:
            dict: Результаты анализа
        """
        # Убедимся, что матрица CSR
        movement_matrix_csr = csr_matrix(self.combined_matrix)

        # Обнулим диагональ (не считаем связи "город-сам_с_собой")
        movement_matrix_csr = movement_matrix_csr.tolil()
        movement_matrix_csr.setdiag(0)
        movement_matrix_csr = movement_matrix_csr.tocsr()

        # Получаем индексы опорных и неопорных городов
        anchor_ids = self.anchor[self.anchor['is_anchor_settlement']].index.tolist()
        non_anchor_ids = self.anchor[~self.anchor['is_anchor_settlement']].index.tolist()

        self.mobility = {}

        # 1. Самообеспеченность и тип города
        self_sufficiency_df, self_sufficiency_pct = self._compute_self_sufficiency(movement_matrix_csr)
        self.mobility['self_sufficiency'] = self_sufficiency_df

        # 2. Покрытие опорными пунктами
        coverage_df = self._compute_anchor_coverage(movement_matrix_csr, anchor_ids, non_anchor_ids)
        self.mobility['anchor_coverage'] = coverage_df

        # 3. Статистика по опорным пунктам
        anchor_stats_df = self._compute_anchor_stats(coverage_df, self_sufficiency_df, anchor_ids)
        self.mobility['anchor_stats'] = anchor_stats_df

        # 4. Потенциальные опорные пункты
        potential_anchors_df = self._compute_potential_anchors(
            movement_matrix_csr, self_sufficiency_pct, non_anchor_ids, anchor_threshold
        )
        self.mobility['potential_anchors'] = potential_anchors_df
        return self.mobility

    def create_map(self, polygons_gdf=None):
        return create_anchor_flow_map(self.combined_matrix, self.anchor, self.mobility, polygons_gdf=polygons_gdf)

