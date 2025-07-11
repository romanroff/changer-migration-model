import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import folium

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import csr_matrix, find

np.random.seed(0)
random.seed(0)

def create_anchor_flow_map(
    matrix: csr_matrix,
    gdf: gpd.GeoDataFrame,
    analysis_results: dict,
    polygons_gdf: gpd.GeoDataFrame = None,
    direction: str = 'index',
    default_tiles: str = 'cartodbpositron'
) -> folium.Map:
    
    if gdf.crs != 'epsg:4326':
        gdf = gdf.copy().to_crs(epsg=4326)
    
    if polygons_gdf is not None:
        if polygons_gdf.crs != 'epsg:4326':
            polygons_gdf = polygons_gdf.copy().to_crs(epsg=4326)
        else:
            polygons_gdf = polygons_gdf.copy()
            
    if direction not in ['index', 'columns']:
        raise ValueError("direction must be either 'index' or 'columns'")
    
    # Убедимся, что gdf использует city_id как индекс
    if 'city_id' not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf['city_id'] = gdf.index
        gdf = gdf.set_index('city_id')

    # Центр карты
    anchor_points = gdf[gdf['is_anchor_settlement']]['geometry']
    center = anchor_points.unary_union.centroid

    # Создаем карту
    fmap = folium.Map(
        location=[center.y, center.x], 
        zoom_start=7,
        tiles=default_tiles,
        attr='Map data © OpenStreetMap contributors'
    )

    # Подложки
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(fmap)
    folium.TileLayer('cartodbdark_matter', name='CartoDB Dark Matter').add_to(fmap)

    # Слои
    edges_layer = folium.FeatureGroup(name='Потоки ≥ 1', show=True)
    edges_below_one = folium.FeatureGroup(name='Потоки < 1', show=False)
    cities_with_edges = folium.FeatureGroup(name='Города с потоками', show=True)
    cities_without_edges = folium.FeatureGroup(name='Города без потоков', show=False)
    polygons_layer = folium.FeatureGroup(name='Полигоны', show=True)

    # Полигоны
    if polygons_gdf is not None:
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)
        for _, row in polygons_gdf.iterrows():
            folium.GeoJson(row['geometry'],
                           style_function=lambda x: {
                               'fillColor': 'blue', 'color': 'black', 'weight': 1, 'fillOpacity': 0.3
                           }).add_to(polygons_layer)

    # Справочники по city_id
    id_to_geom = gdf['geometry'].to_dict()
    id_to_name = gdf['town_name'].to_dict()
    is_anchor = gdf['is_anchor_settlement'].to_dict()

    # Легенда цветовой шкалы
    non_zero_values = matrix.data[matrix.data > 0]
    if len(non_zero_values) == 0:
        raise ValueError("Матрица не содержит ненулевых значений")
        
    min_val, max_val = non_zero_values.min(), non_zero_values.max()
    log_min = np.log10(min_val + 1e-10)
    log_max = np.log10(max_val + 1e-10)
    cmap = plt.get_cmap('RdYlGn_r')
    scalar_map = ScalarMappable(norm=Normalize(vmin=log_min, vmax=log_max), cmap=cmap)

    def get_color(value):
        if value <= 0:
            return "#808080"
        log_val = np.log10(value + 1e-10)
        rgba = scalar_map.to_rgba(log_val)
        return "#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )

    # Tooltip
    self_sufficiency_df = analysis_results['self_sufficiency'].set_index('city_id')

    def create_tooltip(city_id):
        name = id_to_name[city_id]
        row = self_sufficiency_df.loc[city_id]

        tooltip = f"<b>{name}</b><br>"
        tooltip += f"Самообеспеченность: {row['self_sufficiency_pct']:.1f}%<br>"
        tooltip += f"Тип: {row['city_type']}<br>"

        if is_anchor.get(city_id, False):
            tooltip += f"(Опорный пункт)<br>"

            anchor_stats = analysis_results.get('anchor_stats', pd.DataFrame())
            if not anchor_stats.empty:
                stats = anchor_stats[anchor_stats['anchor_id'] == city_id]
                if not stats.empty:
                    stats = stats.iloc[0]
                    if stats['is_weak_anchor']:
                        tooltip += f"<b>Слабый опорный пункт</b><br>"
                    tooltip += f"Средняя обеспеченность других: {stats['mean_coverage']:.1f}%<br>"
                    tooltip += f"Медианная обеспеченность других: {stats['median_coverage']:.1f}%<br>"
                    tooltip += f"Обеспечивает городов: {int(stats['num_covered_cities'])}"
        else:
            pot_df = analysis_results.get('potential_anchors', pd.DataFrame())
            if not pot_df.empty and (pot_df['city_id'] == city_id).any():
                tooltip += f"<b>Потенциальный опорный пункт</b><br>"

        return tooltip

    # Обход разреженной матрицы
    cities_with_edges_set = set()
    flow_data = []

    row, col, data = find(matrix)
    for source, target, value in zip(row, col, data):
        if value <= 0:
            continue

        if direction == 'index':
            from_id, to_id = source, target
        else:
            from_id, to_id = target, source

        cities_with_edges_set.add(from_id)
        cities_with_edges_set.add(to_id)
        flow_data.append((from_id, to_id, value))

    # Отрисовка линий
    for from_id, to_id, value in flow_data:
        from_point = id_to_geom[from_id]
        to_point = id_to_geom[to_id]
        line_color = get_color(value)
        line_weight = 1 + min(np.log1p(abs(value)) / 2, 5)
        tooltip_text = f"{id_to_name[from_id]} → {id_to_name[to_id]}: {value:.2f} чел."

        line = folium.PolyLine(
            locations=[(from_point.y, from_point.x), (to_point.y, to_point.x)],
            color=line_color,
            weight=line_weight,
            opacity=0.8,
            tooltip=tooltip_text,
        )

        if abs(value) < 1:
            line.add_to(edges_below_one)
        else:
            line.add_to(edges_layer)

    # Отрисовка городов
    for city_id in gdf.index:
        point = id_to_geom[city_id]
        marker = folium.CircleMarker(
            location=(point.y, point.x),
            radius=4 if is_anchor[city_id] else 2,
            color="red" if is_anchor[city_id] else "green",
            fill=True,
            fill_opacity=0.9,
            tooltip=create_tooltip(city_id),
            popup=create_tooltip(city_id)
        )
        
        if city_id in cities_with_edges_set:
            marker.add_to(cities_with_edges)
        else:
            marker.add_to(cities_without_edges)

    # Добавляем слои
    polygons_layer.add_to(fmap)
    edges_layer.add_to(fmap)
    edges_below_one.add_to(fmap)
    cities_with_edges.add_to(fmap)
    cities_without_edges.add_to(fmap)

    # Легенда
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 180px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;">
        <b>Легенда (логарифмическая шкала):</b><br>
        <i class="fa fa-circle" style="color:red"></i> Опорные города<br>
        <i class="fa fa-circle" style="color:green"></i> Неопорные города<br>
        <div style="background: linear-gradient(to right, #00ff00, #ff0000); height: 20px;"></div>
        <div style="display: flex; justify-content: space-between;">
            <span>10<sup>{int(log_min)}</sup></span>
            <span>10<sup>{int((log_min+log_max)/2)}</sup></span>
            <span>10<sup>{int(log_max)}</sup></span>
        </div>
        <div style="font-size:12px; color:#555;">Min: {min_val:.0f}, Max: {max_val:.0f}</div>
    </div>
    '''
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Контроль слоев
    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap
