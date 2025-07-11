from functools import singledispatchmethod
import random
from shapely import Point
from townsnet import Region, Provision
from utils.ueqi import UEQI_GROUPS
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