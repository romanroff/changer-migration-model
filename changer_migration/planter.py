import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import pickle

from sklearn.preprocessing import MinMaxScaler
from skmob.models.gravity import Gravity
from shapely.geometry import LineString, Point
from typing import Sequence, Dict, Tuple, Optional, Any
import warnings
import shapely
import skmob

from ._constant import *

def define_model() -> Gravity:
    """
    Создаёт и возвращает гравитационную модель (skmob.models.gravity.Gravity)
    с заранее заданными гиперпараметрами из констант.

    Returns
    -------
    Gravity
        Инициализированная гравити‑модель.
    """
    gravity = Gravity(
        deterrence_func_type="power_law",
        gravity_type="globally constrained",
        destination_exp=DESTINATION_EXP_COEFF,
        origin_exp=ORIGIN_EXP_COEFF,
        deterrence_func_args=[DETERRENCE_FUNC_COEFF],
    )
    return gravity

def drop_cities_no_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет города с отсутствующим или нулевым населением.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица городов. Требуемая колонка: 'population'.

    Returns
    -------
    pd.DataFrame
        Отфильтрованный DataFrame только с population > 0.

    Raises
    ------
    AssertionError
        Если нет колонки 'population'.
    """
    assert "population" in df.columns, "population is not in df.columns"
    mask = df["population"] > 0
    return df.loc[mask].copy()


def normalize_outflow_by_pop_mil(df: pd.DataFrame) -> pd.Series:
    """
    Считает нормированный отток для каждого города:
    migrations_from_each_city * (population / 1e6).

    Parameters
    ----------
    df : pd.DataFrame
        Таблица городов. Требуемые колонки:
        'migrations_from_each_city', 'population'.

    Returns
    -------
    pd.Series
        Серия нормированного оттока, индекс совпадает с df.index.

    Raises
    ------
    AssertionError
        Если нет необходимых колонок.
    """
    POPULATION_NORMALIZATION_VALUE = 1e6
    need = {"migrations_from_each_city", "population"}
    missing = need - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    return df["migrations_from_each_city"] * (df["population"] / POPULATION_NORMALIZATION_VALUE)


def define_scaler() -> MinMaxScaler:
    """
    Возвращает стандартный MinMaxScaler для масштабирования признаков городов.

    Returns
    -------
    MinMaxScaler
        Инициализированный скейлер.
    """
    return MinMaxScaler()


def scale_cities_attrs(
    df: pd.DataFrame,
    cols_to_scale: Sequence[str],
    scaler: MinMaxScaler,
    fit: bool = True,
):
    """
    Масштабирует указанные признаки городов через MinMaxScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с признаками. Должна содержать колонки из cols_to_scale.
        Допускается передавать и одно наблюдение (одна строка).
    cols_to_scale : Sequence[str]
        Список имён колонок для масштабирования.
    scaler : MinMaxScaler
        Объект скейлера. Если fit=True, будет обучён на df[cols_to_scale].
    fit : bool, default True
        Обучать ли скейлер перед трансформацией.

    Returns
    -------
    numpy.ndarray
        Масштабированная матрица признаков той же формы, что df[cols_to_scale].

    Notes
    -----
    Если по ошибке признаки оказались в индексе (а не в колонках),
    функция попробует транспонировать df.
    """
    # Если признаки оказались в индексе — повернём таблицу
    if set(cols_to_scale).issubset(df.index) and not set(cols_to_scale).issubset(df.columns):
        df = df.T

    X = df.loc[:, list(cols_to_scale)]
    return scaler.fit_transform(X) if fit else scaler.transform(X)


def calculate_attractiveness(df: pd.DataFrame) -> pd.Series:
    """
    Векторно вычисляет коэффициент привлекательности города по формуле:
      FACTORY_SALARY_W_COEFF * (factories_total + median_salary) * CITY_PARAMS_W_COEFF
      + сумма UEQI-показателей
      + (1 - harsh_climate) + 1

    Параметры взвешивания берутся из констант. Возвращает целочисленную Series.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица городов с нужными колонками:
        'factories_total', 'median_salary',
        'ueqi_residential', 'ueqi_street_networks', 'ueqi_green_spaces',
        'ueqi_public_and_business_infrastructure',
        'ueqi_social_and_leisure_infrastructure', 'ueqi_citywide_space',
        'harsh_climate'.

    Returns
    -------
    pd.Series
        Series коэффициента привлекательности (int), выровненная по индексу df.

    Raises
    ------
    AssertionError
        Если отсутствуют необходимые колонки.
    """
    required = {
        "factories_total", "median_salary",
        "ueqi_residential", "ueqi_street_networks", "ueqi_green_spaces",
        "ueqi_public_and_business_infrastructure",
        "ueqi_social_and_leisure_infrastructure", "ueqi_citywide_space",
        "harsh_climate",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    ueqi_sum = (
        df["ueqi_residential"] + df["ueqi_street_networks"] + df["ueqi_green_spaces"] +
        df["ueqi_public_and_business_infrastructure"] +
        df["ueqi_social_and_leisure_infrastructure"] + df["ueqi_citywide_space"]
    )

    score = (
        FACTORY_SALARY_W_COEFF * (df["factories_total"] + df["median_salary"]) * CITY_PARAMS_W_COEFF
        + ueqi_sum
        + (1 - df["harsh_climate"])
        + 1
    )

    return score.round(0).astype(int)

def filter_od_matrix_resetted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрует OD-таблицу, убирая строки, где origin == destination.

    Parameters
    ----------
    df : pd.DataFrame
        Длинная OD-таблица с колонками 'origin', 'destination' (и, опционально, 'flow').

    Returns
    -------
    pd.DataFrame
        Подтаблица только с парами (origin != destination).

    Raises
    ------
    AssertionError
        Если отсутствуют колонки 'origin' или 'destination'.
    """
    need = {"origin", "destination"}
    missing = need - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    mask = df["origin"] != df["destination"]
    return df.loc[mask].copy()


def reset_od_matrix(od_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует широкую OD-матрицу в длинный формат (origin, destination, flow).

    Ожидается, что индекс после reset_index() содержит колонку 'region_city',
    которая интерпретируется как 'origin', а остальные колонки — назначения.

    Parameters
    ----------
    od_matrix : pd.DataFrame
        Широкая матрица OD: индекс — города (region_city), колонки — города‑назначения, значения — потоки.

    Returns
    -------
    pd.DataFrame
        Длинная таблица с колонками: 'origin', 'destination', 'flow'.
    """
    od_matrix_reset = od_matrix.reset_index()
    assert "region_city" in od_matrix_reset.columns, "Expected 'region_city' after reset_index()"
    od_matrix_reset = od_matrix_reset.rename(columns={"region_city": "origin"})
    long_df = pd.melt(
        od_matrix_reset, id_vars=["origin"], var_name="destination", value_name="flow"
    )
    return long_df


def check_filter_cities_in_od_matrix(df: pd.DataFrame, od_df: pd.DataFrame) -> pd.DataFrame:
    """
    Оставляет в таблице городов только те записи, которые встречаются в OD-таблице
    (либо как origin, либо как destination).

    Parameters
    ----------
    df : pd.DataFrame
        Таблица городов, требуется колонка 'region_city'.
    od_df : pd.DataFrame
        Длинная OD-таблица с колонками 'origin' и 'destination'.

    Returns
    -------
    pd.DataFrame
        Отфильтрованный df с городами, присутствующими в od_df.

    Raises
    ------
    AssertionError
        Если отсутствуют необходимые колонки.
    """
    assert "region_city" in df.columns, "'region_city' not in cities DataFrame"
    need = {"origin", "destination"}
    missing = need - set(od_df.columns)
    assert not missing, f"Missing OD columns: {missing}"
    present = set(od_df["origin"].tolist() + od_df["destination"].tolist())
    return df.loc[df["region_city"].isin(present)].copy()


def make_flow_df(od_df: pd.DataFrame, df_with_od_geoms: gpd.GeoDataFrame) -> skmob.FlowDataFrame:
    """
    Создаёт skmob.FlowDataFrame из длинной OD‑таблицы и тесcеляции (геометрий городов).

    Parameters
    ----------
    od_df : pd.DataFrame
        Длинная OD‑таблица с колонками 'origin', 'destination', 'flow'.
    df_with_od_geoms : gpd.GeoDataFrame
        Геометрии городов (тесселяция) с колонкой 'region_city' и 'geometry'.

    Returns
    -------
    skmob.FlowDataFrame
        Объект потоков для skmob.
    """
    fdf = skmob.FlowDataFrame(
        data=od_df,
        origin="origin",
        destination="destination",
        flow="flow",
        tessellation=df_with_od_geoms,
        tile_id="region_city",
    )
    return fdf


def fit_flow_df(fdf: skmob.FlowDataFrame, gravity: Gravity) -> None:
    """
    Обучает гравитационную модель на FlowDataFrame.

    Parameters
    ----------
    fdf : skmob.FlowDataFrame
        Потоки с атрибутами узлов (должен содержать колонку 'city_attractiveness_coeff' в тесселяции).
    gravity : Gravity
        Инициализированная гравитационная модель skmob.
    """
    gravity.fit(fdf, relevance_column="city_attractiveness_coeff")


def generate_flows(df: pd.DataFrame, gravity: Gravity) -> pd.DataFrame:
    """
    Генерирует таблицу потоков (origin, destination, flow) по городам
    на основе обученной гравитационной модели.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с узлами (городами), должна содержать:
        'city_attractiveness_coeff', 'region_city', 'norm_outflow'.
    gravity : Gravity
        Обученная модель гравитации (после fit()).

    Returns
    -------
    pd.DataFrame
        Таблица потоков со столбцами: 'origin', 'destination', 'flow', отсортированная по ['flow', 'destination'].

    Raises
    ------
    AssertionError
        Если отсутствуют необходимые колонки в df.
    """
    need = {"city_attractiveness_coeff", "region_city", "norm_outflow"}
    missing = need - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    fdf = gravity.generate(
        df,
        relevance_column="city_attractiveness_coeff",
        tot_outflows_column="norm_outflow",
        out_format="flows",
        tile_id_column="region_city",
    )
    return pd.DataFrame(fdf).sort_values(by=["flow", "destination"]).reset_index(drop=True)


def inverse_scale_df(df: pd.DataFrame, cols: Sequence[str], scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Обратное преобразование масштабированных признаков (inverse_transform).

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с масштабированными признаками.
    cols : Sequence[str]
        Список колонок для обратного преобразования.
    scaler : MinMaxScaler
        Обученный MinMaxScaler.

    Returns
    -------
    pd.DataFrame
        Таблица с восстановленными значениями только по колонкам cols.
    """
    X = df.loc[:, list(cols)]
    inv = scaler.inverse_transform(X)
    return pd.DataFrame(inv, index=df.index, columns=list(cols))


def tailor_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Снижает «точность» координат геометрий (snap-to-grid) для стабильных топологий
    и уменьшения шумов при операциях наложения.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Геодатафрейм с колонкой 'geometry'. CRS сохраняется.

    Returns
    -------
    gpd.GeoSeries
        Геосерия с скорректированной точностью координат (grid size = 0.001).

    Notes
    -----
    Размер сетки GRID_SIZE=0.001 приблизительно соответствует ~1e-3 в единицах CRS,
    для географических CRS (~градусы) это около сотен метров по широте.
    """
    GRID_SIZE = 0.001
    return gpd.GeoSeries(
        shapely.set_precision(gdf["geometry"].array, grid_size=GRID_SIZE),
        index=gdf.index,
        crs=gdf.crs,
        name="geometry",
    )


def post_processing(gdf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Считает расстояние для линий OD, формирует маску «валидных» связей
    и производные метрики для визуализации.

    Маска отбрасывает:
      • слишком длинные линии (>= DISTANCE_TRASHOLD_METERS),
      • слишком маленькие потоки (<= MINIMAL_FLOW),
      • аномально большие потоки (>= MAXIMAL_FLOW).

    Parameters
    ----------
    gdf : pd.DataFrame
        Таблица связей с колонками:
        'geometry' (линии OD в CRS DEGREE_CRS), 'flow' (число).

    Returns
    -------
    (pd.DataFrame, pd.Series)
        Модифицированный gdf с колонками:
          - 'distance' (метры, в CRS METRIC_CRS),
          - 'big_flows' (инт. усиление для визуализации),
          - 'scaled_flows_forvis' (скейл ширины линий для карт),
        и булева маска валидных записей той же длины.

    Raises
    ------
    AssertionError
        Если отсутствуют колонки 'geometry' или 'flow'.
    """
    assert all(c in gdf.columns for c in ("geometry", "flow")), "Missing 'geometry' or 'flow'"

    MINIMAL_FLOW = 1e-3   # всё МЕНЬШЕ или РАВНО этому считаем шумом
    MAXIMAL_FLOW = 4      # всё БОЛЬШЕ или РАВНО этому считаем шумом

    # расстояние линий в метрах
    gdf = gdf.copy()
    gdf["distance"] = (
        gpd.GeoSeries(gdf["geometry"], crs=DEGREE_CRS)
        .to_crs(METRIC_CRS)
        .length
    )

    mask_valid = (
        (gdf["distance"] < DISTANCE_TRASHOLD_METERS)
        & (gdf["flow"] > MINIMAL_FLOW)
        & (gdf["flow"] < MAXIMAL_FLOW)
    )

    # эмпирическое усиление для выразительности на карте
    gdf.loc[:, "big_flows"] = (np.log(gdf["flow"] + 5.0) ** 10).astype(int)
    gdf.loc[:, "scaled_flows_forvis"] = np.round(np.log(gdf["big_flows"]) / 1e2, 3)

    return gdf, mask_valid

def join_od_params(fdf_with_flows: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """
    Обогащает таблицу потоков атрибутами городов-источников и городов-приёмников.

    Parameters
    ----------
    fdf_with_flows : pd.DataFrame
        Таблица потоков с колонками: 'origin', 'destination', 'flow'.
    cities : pd.DataFrame
        Таблица городов с колонками:
        'region_city', 'city_attractiveness_coeff', 'population'.

    Returns
    -------
    pd.DataFrame
        fdf_with_flows + атрибуты:
         - destination_attr (привлекательность приёмника),
         - origin_attr (привлекательность источника),
         - population (население приёмника).
    """
    need_flows = {"origin", "destination", "flow"}
    need_cities = {"region_city", "city_attractiveness_coeff", "population"}
    miss_f = need_flows - set(fdf_with_flows.columns)
    miss_c = need_cities - set(cities.columns)
    assert not miss_f, f"Missing in flows: {miss_f}"
    assert not miss_c, f"Missing in cities: {miss_c}"

    dest = cities[["region_city", "city_attractiveness_coeff", "population"]].rename(
        columns={"region_city": "destination", "city_attractiveness_coeff": "destination_attr"}
    )
    orig = cities[["region_city", "city_attractiveness_coeff"]].rename(
        columns={"region_city": "origin", "city_attractiveness_coeff": "origin_attr"}
    )

    df_links = (
        fdf_with_flows.merge(dest, on="destination", how="left")
                      .merge(orig, on="origin", how="left")
    )
    return df_links


# Define the function that uses the pre-constructed dictionary
def create_linestring(row: pd.Series, geometry_dict: Dict[str, Point]) -> Optional[LineString]:
    """
    Создаёт линию OD между центрами городов.

    Parameters
    ----------
    row : pd.Series
        Строка из таблицы потоков; ожидает поля 'origin', 'destination'.
    geometry_dict : dict[str, shapely.geometry.Point]
        Словарь {'region_city': Point} с координатами городов.

    Returns
    -------
    LineString | None
        Линия между точками (origin -> destination) или None, если
        хотя бы одна из точек отсутствует.
    """
    o = row["origin"]
    d = row["destination"]
    if o in geometry_dict and d in geometry_dict:
        return LineString([geometry_dict[o], geometry_dict[d]])
    return None


def make_od_linestring_geom(fdf_fitted_df: pd.DataFrame, init_cities: gpd.GeoDataFrame) -> pd.Series:
    """
    Строит геометрию линий OD по списку потоков и геометриям городов.

    Parameters
    ----------
    fdf_fitted_df : pd.DataFrame
        Таблица потоков с колонками 'origin', 'destination'.
    init_cities : gpd.GeoDataFrame
        Геометрии городов (точки центров) с колонкой 'region_city' и 'geometry'.

    Returns
    -------
    pd.Series
        Серия геометрий LineString (совместима с присваиванием в df['geometry']).

    """
    assert {"origin", "destination"}.issubset(fdf_fitted_df.columns), "Missing 'origin'/'destination' in flows"
    assert {"region_city", "geometry"}.issubset(init_cities.columns), "Missing 'region_city'/'geometry' in cities"

    city_geometry_dict: Dict[str, Point] = init_cities.set_index("region_city")["geometry"].to_dict()
    return fdf_fitted_df.apply(lambda row: create_linestring(row, city_geometry_dict), axis=1)


def make_folium_map(gdf_links, cities, region_poly=None):
    assert all(
        attr in gdf_links.columns
        for attr in [
            "geometry",
            "scaled_flows_forvis",
            "origin",
            "destination",
            "big_flows",
        ]
    )

    assert isinstance(gdf_links, gpd.GeoDataFrame)
    assert isinstance(cities, gpd.GeoDataFrame)

    m = gdf_links[
        ["geometry", "scaled_flows_forvis", "origin", "destination", "big_flows"]
    ].explore(
        scheme="Percentiles",
        column="big_flows",
        cmap="Accent_r",
        style_kwds={
            "style_function": lambda feature: {
                "weight": (
                    feature["properties"]["scaled_flows_forvis"] + 1
                ),  # Set line width based on the attribute
                "opacity": 0.3,  # Adjust opacity if necessary
            }
        },
        control_scale=True,
        vmin=10,
        vmax=2.5e2,
        tiles="Cartodb dark_matter",
    )

    # Create a style function for circle markers
    def style_function(x, min_radius=1, max_radius=10):

        # Get the value for the chosen parameter
        flows_in_value = x["flows_in"]  # Default to 1 to avoid log(0) errors
        flows_out_value = x["flows_out"]  # Default to 1 to avoid log(0) errors

        # Compute the logarithmic value (base 10 or natural log)
        log_flows_in_value = np.sqrt(
            flows_in_value
        )  # Natural logarithm, you can use np.log10() for base 10

        # Compute the logarithmic value (base 10 or natural log)
        log_flows_out_value = np.sqrt(
            flows_out_value
        )  # Natural logarithm, you can use np.log10() for base 10

        # Normalize the log value to adjust the circle radius
        # Ensure the log value is scaled between min_radius and max_radius
        # marker_radius = min(max(flows_in_value, min_radius), max_radius)
        # border_radius = min(max(flows_out_value, min_radius), max_radius)

        return folium.CircleMarker(
            location=[x["geometry"].y, x["geometry"].x],
            radius=flows_in_value / 500,  # Adjust radius as needed
            # weight=log_flows_out_value / 5,
            popup=x[["region_city", "flows_in", "flows_out"]],
            fill=True,
            # fill_color="white",
            weight=1,
            color="white",
            opacity=1,  # Set border opacity
            fill_color="black",
            fill_opacity=0.01,
        ).add_to(m)

    # Apply the function to each feature in GeoJson
    cities.apply(lambda row: style_function(row), axis=1)

    # Create an HTML title element
    title_text = "Все потоки без разделения по профессиям"
    title_html = f"""
        <div style="
            position: fixed; 
            top: 10%;  
            left: 20%; 
            transform: translateX(-50%);
            background-color: transparent; 
            color: white; 
            font-size: 20px; 
            font-weight: bold;
            z-index: 1000;">
            {title_text}
        </div>
        """

    if isinstance(region_poly, gpd.GeoDataFrame):

        # Add the GeoDataFrame as a GeoJSON layer with borders only
        folium.GeoJson(
            region_poly.geometry.item().boundary,
            name="geojson",
            style_function=lambda feature: {
                "fillOpacity": 0,  # No fill
                # "fill_color": "yellow",
                "color": "white",  # Border color
                "weight": 0.4,  # Border thickness
            },
        ).add_to(m)

    # Add the title element to the map
    m.get_root().html.add_child(folium.Element(title_html))
    # Add layer control to toggle GeoJSON layer visibility
    folium.LayerControl().add_to(m)

    return m

class WorkForceFlows:
    def __init__(self):
        self.cols = [
            "population",
            "harsh_climate",
            "ueqi_residential",
            "ueqi_street_networks",
            "ueqi_green_spaces",
            "ueqi_public_and_business_infrastructure",
            "ueqi_social_and_leisure_infrastructure",
            "ueqi_citywide_space",
            "median_salary",
            "factories_total",
        ]
        self.cols_to_round = ["city_attractiveness_coeff", "population"]

        # Flag for tracking which stages need recalculation
        self.pipeline_stages = {
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
        }

        # Track initial state of cities
        self.initial_cities_state = None
        self.initial_links_state = None
        self.prev_cities_state = None  # Track the last saved state (for comparison)
        self.prev_links_state = None
        self.current_cities_state = None  # Track the current state of cities
        self.current_links_state = None
        self.scaled_cities = None
        self.update_city_name = None
        self.update_city_name_idx = None
        self.updated_city_params = None
        self.fdf = None
        self.od_linestrings = None

    def __getitem__(self, key: str) -> Any:
        """
        Достаёт атрибут экземпляра по ключу (как из словаря).

        Parameters
        ----------
        key : str
            Имя атрибута.

        Returns
        -------
        Any
            Значение атрибута или диагностическая строка, если атрибут не найден.
        """
        return getattr(self, key, f"Property '{key}' not found")


    def __setitem__(self, key: str, value: Any) -> None:
        """
        Устанавливает атрибут экземпляра по ключу (как в словаре).

        Parameters
        ----------
        key : str
            Имя атрибута.
        value : Any
            Новое значение атрибута.
        """
        if hasattr(self, key):
            print("Warning: rewriting existing attribute")
        setattr(self, key, value)


    def save_initial_state(self) -> None:
        """
        Сохраняет исходное состояние городов и связей (первый снимок).
        Вызывается один раз после первой полной сборки графа.
        """
        if self.initial_cities_state is None:
            self.initial_cities_state = self.cities.copy()
            self.initial_links_state = self.gdf_links.copy()
            print("Initial cities state saved.")


    def save_previous_state(self) -> None:
        """
        Сохраняет предыдущее состояние (до обновления параметров).
        Полезно для отката/сравнения.
        """
        self.prev_cities_state = self.cities.copy()
        self.prev_links_state = self.gdf_links.copy()


    def save_current_state(self) -> None:
        """
        Сохраняет текущее состояние (после обновления/пересчёта),
        чтобы затем сравнить с initial_state.
        """
        self.current_cities_state = self.cities.copy()
        self.current_links_state = self.gdf_links.copy()


    def compare_city_states(self) -> gpd.GeoDataFrame | bool:
        """
        Сравнивает текущее состояние городов с исходным по показателям
        'flows_in'/'flows_out' и возвращает пространственную таблицу
        с разностями.

        Returns
        -------
        gpd.GeoDataFrame | bool
            GeoDataFrame с колонками:
            ['region_city', 'geometry', 'in_out_diff', 'in_diff', 'out_diff']
            в CRS=DEGREE_CRS, либо False, если отсутствуют состояния.
        """
        if hasattr(self, "current_cities_state") and hasattr(self, "initial_cities_state"):
            diff_cities = self.current_cities_state[
                ["flows_in", "flows_out", "region_city", "geometry"]
            ].merge(
                self.initial_cities_state[["flows_in", "flows_out", "region_city"]]
                .rename(columns={"flows_in": "flows_in_prev", "flows_out": "flows_out_prev"}),
                on="region_city",
                how="left",
            )

            diff_cities["in_diff"] = diff_cities["flows_in"] - diff_cities["flows_in_prev"]
            diff_cities["out_diff"] = diff_cities["flows_out"] - diff_cities["flows_out_prev"]
            diff_cities["in_out_diff"] = diff_cities["in_diff"] - diff_cities["out_diff"]

            # отсечь мелкие флуктуации
            threshold = 3
            mask_fluctuation = diff_cities["in_out_diff"].abs() <= threshold
            diff_cities.loc[mask_fluctuation, "in_out_diff"] = 0

            gdf = gpd.GeoDataFrame(
                diff_cities[["region_city", "geometry", "in_out_diff", "in_diff", "out_diff"]],
                geometry="geometry",
                crs=DEGREE_CRS,
            )
            return gdf
        else:
            print("Both states must be DataFrame objects.")
            return False


    def compare_link_states(self) -> pd.DataFrame | bool:
        """
        Сравнивает текущие связи (рёбра) с исходными по метрике 'big_flows'
        и возвращает таблицу разностей.

        Returns
        -------
        pd.DataFrame | bool
            DataFrame с колонками связей и обновлённым 'big_flows'
            (это уже разность с исходным), либо False, если нет состояний.
        """
        if hasattr(self, "current_cities_state") and hasattr(self, "initial_links_state"):
            diff_links = (
                self.initial_links_state[
                    ["origin", "destination", "big_flows", "geometry", "scaled_flows_forvis"]
                ]
                .rename(columns={"big_flows": "init_flows"})
                .merge(self.current_links_state[["origin", "destination", "big_flows"]], on=["origin", "destination"], how="left")
            )

            diff_links["big_flows"] = diff_links["big_flows"] - diff_links["init_flows"]

            threshold = 3
            mask_fluctuation = diff_links["big_flows"].abs() <= threshold
            diff_links.loc[mask_fluctuation, "big_flows"] = 0

            return diff_links.drop(columns=["init_flows"])
        else:
            print("Both states must be DataFrame objects.")
            return False


    def reset_state(self) -> None:
        """
        Возвращает self.cities к исходному (initial) состоянию.
        """
        if self.initial_cities_state is not None:
            self.cities = self.initial_cities_state.copy()
            print("Cities state reset to the initial state.")
        else:
            print("No initial state to reset to.")


    @classmethod
    def make_scaler(cls) -> None:
        """
        Инициализирует и сохраняет скейлер признаков на уровне класса.
        """
        cls.scaler = define_scaler()

    @classmethod
    def make_model(cls) -> None:
        """
        Инициализирует и сохраняет гравитационную модель на уровне класса.
        """
        cls.model = define_model()


    def mark_stage_dirty(self, stage_number: int) -> None:
        """
        Помечает указанную стадию и все последующие как «требуют пересчёта».

        Parameters
        ----------
        stage_number : int
            Номер стадии (1..8), с которой начинать инвалидировать пайплайн.
        """
        for stage in range(stage_number, max(self.pipeline_stages.keys()) + 1):
            self.pipeline_stages[stage] = False


    def run_cities_pipeline_stage_1(self) -> None:
        """
        Stage 1: фильтрация городов (population>0) и расчёт нормированного оттока.
        Требует: self.cities.
        """
        if not self.pipeline_stages[1]:
            if hasattr(self, "cities"):
                self.cities = drop_cities_no_population(self.cities)
                self.cities["norm_outflow"] = normalize_outflow_by_pop_mil(self.cities)
                self.pipeline_stages[1] = True
                self.mark_stage_dirty(2)
            else:
                warnings.warn("Please provide 'cities' data")
        else:
            print("Skipping: Stage 1 has already been run")


    def run_cities_pipeline_stage_2(self) -> None:
        """
        Stage 2: подготовка init_cities в CRS DEGREE_CRS и стабилизация геометрий.
        Требует: self.cities.
        """
        if not self.pipeline_stages[2]:
            if hasattr(self, "cities"):
                self.init_cities = self.cities.copy().to_crs(DEGREE_CRS)
                self.init_cities["geometry"] = tailor_geometries(self.init_cities)
                self.pipeline_stages[2] = True
                self.mark_stage_dirty(3)
            else:
                warnings.warn("Please provide 'cities' data")
        else:
            print("Skipping: Stage 2 has already been run")


    def run_cities_pipeline_stage_3(self) -> None:
        """
        Stage 3: преобразование OD-матрицы в длинный формат и фильтрация self‑петель.
        Требует: self.cities, self.od.
        """
        if not self.pipeline_stages[3]:
            if hasattr(self, "cities") and hasattr(self, "od"):
                self.od_matrix_reset = reset_od_matrix(self["od"])
                self.od_matrix_reset = filter_od_matrix_resetted(self.od_matrix_reset)
                self.od_matrix_reset.reset_index(drop=True, inplace=True)
                self.pipeline_stages[3] = True
                self.mark_stage_dirty(4)
            else:
                warnings.warn("Please provide 'cities' and 'od' data")
        else:
            print("Skipping: Stage 3 has already been run")


    def run_cities_pipeline_stage_4(self) -> None:
        """
        Stage 4: масштабирование признаков и расчёт attractiveness, перенос в init_cities.
        Требует: self.cities, self.scaler.
        """
        if not self.pipeline_stages[4]:
            if hasattr(self, "cities") and hasattr(self, "scaler"):
                self.cities.loc[:, self.cols] = scale_cities_attrs(
                    self.cities, self.cols, self.scaler, fit=True
                )
                self.scaled_cities = self.cities.copy()
                self.cities["city_attractiveness_coeff"] = calculate_attractiveness(self.cities)
                self.init_cities["city_attractiveness_coeff"] = self.cities["city_attractiveness_coeff"].copy()
                self.pipeline_stages[4] = True
                self.mark_stage_dirty(5)
            else:
                warnings.warn("Please provide 'cities' data and a scaler")
        else:
            print("Skipping: Stage 4 has already been run")


    def run_cities_pipeline_stage_5(self) -> None:
        """
        Stage 5: создание FlowDataFrame и обратное масштабирование признаков для хранения/вывода.
        Требует: self.cities, self.od_matrix_reset.
        """
        if not self.pipeline_stages[5]:
            if hasattr(self, "cities") and hasattr(self, "od_matrix_reset"):
                self.fdf = make_flow_df(self.od_matrix_reset, self.cities)
                self.cities.loc[:, self.cols] = inverse_scale_df(self.cities, self.cols, self.scaler)
                self.cities.loc[:, self.cols_to_round] = self.cities.loc[:, self.cols_to_round].astype(int)
                self.pipeline_stages[5] = True
                self.mark_stage_dirty(6)
            else:
                warnings.warn("Please provide 'cities' and 'od_matrix_reset' data")
        else:
            print("Skipping: Stage 5 has already been run")


    def run_cities_pipeline_stage_6(self) -> None:
        """
        Stage 6: фит гравитационной модели и генерация потоков.
        Требует: self.fdf, self.init_cities, self.model.
        """
        if not self.pipeline_stages[6]:
            if hasattr(self, "fdf") and hasattr(self, "init_cities"):
                fit_flow_df(self.fdf, self.model)
                self.fdf_fitted_df = generate_flows(self.cities, self.model)
                self.pipeline_stages[6] = True
                self.mark_stage_dirty(7)
            else:
                warnings.warn("Please provide 'fdf' and 'init_cities' data")
        else:
            print("Skipping: Stage 6 has already been run")


    def run_cities_pipeline_stage_7(self) -> None:
        """
        Stage 7: построение геометрий линий OD и запись их в таблицу потоков.
        Требует: self.fdf_fitted_df, self.init_cities.
        """
        if not self.pipeline_stages[7]:
            if hasattr(self, "fdf_fitted_df") and hasattr(self, "init_cities"):
                self.od_linestrings = make_od_linestring_geom(self.fdf_fitted_df, self.init_cities)
                self.fdf_fitted_df["geometry"] = self.od_linestrings
                self.pipeline_stages[7] = True
                self.mark_stage_dirty(8)
            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'init_cities' data")
        else:
            print("Skipping: Stage 7 has already been run")


    def run_cities_pipeline_stage_8(self) -> None:
        """
        Stage 8: обогащение потоков атрибутами городов, пост‑обработка, сбор агрегатов flows_in/out.
        Требует: self.fdf_fitted_df, self.cities.
        """
        if not self.pipeline_stages[8]:
            if hasattr(self, "fdf_fitted_df") and hasattr(self, "init_cities"):
                self.df_links = join_od_params(self.fdf_fitted_df, self.cities)
                self.df_links, self.mask_distance_flow = post_processing(self.df_links)

                self.gdf_links = gpd.GeoDataFrame(
                    self.df_links[self.mask_distance_flow], crs=DEGREE_CRS
                )
                self.gdf_links["geometry"] = tailor_geometries(self.gdf_links)
                self.pipeline_stages[8] = True

                flows_grouped_out = (
                    self.gdf_links.drop(columns=["destination", "geometry"])
                    .groupby("origin").sum().reset_index(drop=False)
                    .loc[:, ["origin", "big_flows"]]
                    .rename(columns={"big_flows": "flows_out", "origin": "region_city"})
                )

                flows_grouped_in = (
                    self.gdf_links.drop(columns=["origin", "geometry"])
                    .groupby("destination").sum().reset_index(drop=False)
                    .loc[:, ["destination", "big_flows"]]
                    .rename(columns={"big_flows": "flows_in", "destination": "region_city"})
                )

                self.cities = self.cities.merge(flows_grouped_in, how="left").merge(flows_grouped_out, how="left")

                self.save_initial_state()
                self.save_current_state()
            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'cities' data")
        else:
            print("Skipping: Stage 8 has already been run")


    # -----------------------------------------------------------------
    def run_cities_pipeline_stage_4_upd(self) -> None:
        """
        Stage 4 (update): локальный пересчёт для одного города (рескейл признаков и attractiveness).
        Требует: self.cities, self.scaler, self.update_city_name/_idx, self.scaled_cities.
        """
        if not self.pipeline_stages[4]:
            if hasattr(self, "cities") and hasattr(self, "scaler"):
                if self.update_city_name:
                    self.scaled_cities.loc[self.update_city_name_idx, self.cols] = (
                        scale_cities_attrs(
                            self.cities.loc[self.update_city_name_idx, :].to_frame(),
                            self.cols,
                            self.scaler,
                            fit=False,
                        )
                    )
                    self.cities.loc[self.update_city_name_idx, "city_attractiveness_coeff"] = (
                        calculate_attractiveness(
                            self.scaled_cities.loc[self.update_city_name_idx, :].to_frame().T
                        ).item()
                    )
                    self.pipeline_stages[4] = True
                    self.mark_stage_dirty(5)
            else:
                warnings.warn("Please provide 'cities' data and a scaler")
        else:
            print("Skipping: Stage 4 has already been run")

    def run_cities_pipeline_stage_5_upd(self) -> None:
        """
        Stage 5 (update): округление пересчитанных целевых колонок для одного города.
        Требует: self.cities, self.od_matrix_reset.
        """
        if not self.pipeline_stages[5]:
            if hasattr(self, "cities") and hasattr(self, "od_matrix_reset"):
                self.cities.loc[self.update_city_name_idx, self.cols_to_round] = (
                    self.cities.loc[self.update_city_name_idx, self.cols_to_round].astype(int)
                )
                self.pipeline_stages[5] = True
                self.mark_stage_dirty(6)
            else:
                warnings.warn("Please provide 'cities' and 'od_matrix_reset' data")
        else:
            print("Skipping: Stage 5 has already been run")


    def run_cities_pipeline_stage_6_upd(self) -> None:
        """
        Stage 6 (update): генерация потоков без повторного fit модели.
        Требует: self.fdf, self.init_cities.
        """
        if not self.pipeline_stages[6]:
            if hasattr(self, "fdf") and hasattr(self, "init_cities"):
                self.fdf_fitted_df = generate_flows(self.cities, self.model)
                self.pipeline_stages[6] = True
                self.mark_stage_dirty(7)
            else:
                warnings.warn("Please provide 'fdf' and 'init_cities' data")
        else:
            print("Skipping: Stage 6 has already been run")


    def run_cities_pipeline_stage_7_upd(self) -> None:
        """
        Stage 7 (update): обновление геометрий линий для уже рассчитанных потоков.
        Требует: self.fdf_fitted_df, self.od_linestrings.
        """
        if not self.pipeline_stages[7]:
            fitted_df = getattr(self, "fdf_fitted_df", None)
            if fitted_df is not None:
                fitted_df["geometry"] = self.od_linestrings
                self.pipeline_stages[7] = True
                self.mark_stage_dirty(8)
            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'init_cities' data")
        else:
            print("Skipping: Stage 7 has already been run")


    def run_cities_pipeline_stage_8_upd(self) -> None:
        """
        Stage 8 (update): обогащение, пост‑обработка и пересбор агрегатов flows_in/out
        после локального обновления.
        """
        if not self.pipeline_stages[8]:
            fitted_df = getattr(self, "fdf_fitted_df", None)
            cities_df = getattr(self, "cities", None)

            if fitted_df is not None and cities_df is not None:
                self.df_links = join_od_params(fitted_df, cities_df)
                self.df_links, self.mask_distance_flow = post_processing(self.df_links)

                self.gdf_links = gpd.GeoDataFrame(
                    self.df_links.loc[self.mask_distance_flow], crs=DEGREE_CRS
                )
                self.gdf_links["geometry"] = tailor_geometries(self.gdf_links)
                self.pipeline_stages[8] = True

                flows_out = (
                    self.gdf_links.drop(columns=["destination", "geometry"])
                    .groupby("origin", as_index=False)["big_flows"]
                    .sum()
                    .rename(columns={"big_flows": "flows_out", "origin": "region_city"})
                )

                if "flows_in" in cities_df.columns:
                    cities_df.drop(columns=["flows_in", "flows_out"], inplace=True)

                flows_in = (
                    self.gdf_links.drop(columns=["origin", "geometry"])
                    .groupby("destination", as_index=False)["big_flows"]
                    .sum()
                    .rename(columns={"big_flows": "flows_in", "destination": "region_city"})
                )

                self.cities = cities_df.merge(flows_in, how="left").merge(flows_out, how="left")
                self.save_current_state()
            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'cities' data")
        else:
            print("Skipping: Stage 8 has already been run")


    # -----------------------------------------------------------------

    def update_city_params(self, city_name: str, new_params: Dict[str, Any]) -> None:
        """
        Обновляет параметры одного города и помечает стадии для пересчёта.

        Parameters
        ----------
        city_name : str
            Имя города (значение в колонке 'region_city').
        new_params : dict[str, Any]
            Словарь {колонка: новое_значение}.
        """
        self.save_previous_state()

        self.update_city_name = city_name
        self.updated_city_params = new_params

        if city_name in self.cities["region_city"].values:
            self.update_city_name_idx = self.cities[self.cities["region_city"] == city_name].index.item()
            self.cities.loc[self.update_city_name_idx, new_params.keys()] = new_params.values()
            print(f"Updated parameters for {city_name}")
            self.mark_stage_dirty(4)
        else:
            print(f"City {city_name} not found in the DataFrame.")


    def recalculate_after_update(self) -> None:
        """
        Пересчитывает стадии пайплайна 4→8 после обновления параметров одного города.
        """
        print("Recalculating after updating parameters")
        self.run_cities_pipeline_stage_4_upd()
        self.run_cities_pipeline_stage_5_upd()
        self.run_cities_pipeline_stage_6_upd()
        self.run_cities_pipeline_stage_7_upd()
        self.run_cities_pipeline_stage_8_upd()
        print("Recalculation complete.")


    def to_pickle(self, filename: str) -> None:
        """
        Сохраняет весь инстанс класса в pickle.

        Parameters
        ----------
        filename : str
            Путь к файлу *.pkl.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Class instance saved to {filename}")


    @classmethod
    def from_pickle(cls, filename: str) -> "WorkForceFlows":
        """
        Загружает инстанс класса из pickle.

        Parameters
        ----------
        filename : str
            Путь к файлу *.pkl.

        Returns
        -------
        WorkForceFlows
            Восстановленный экземпляр.
        """
        with open(filename, "rb") as f:
            instance = pickle.load(f)
        print(f"Class instance loaded from {filename}")
        return instance

