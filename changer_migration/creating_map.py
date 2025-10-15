import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import LineString
from matplotlib.collections import LineCollection

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import csr_matrix, find


def create_anchor_flow_map(
    matrix: csr_matrix,
    gdf: gpd.GeoDataFrame,
    analysis_results: dict,
    polygons_gdf: gpd.GeoDataFrame = None,
    direction: str = 'index',
    default_tiles: str = 'cartodbpositron',
    *,
    group_mobility: dict | None = None,
    group_matrices: dict | None = None,
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
    folium.TileLayer('cartodbdark_matter',
                     name='CartoDB Dark Matter').add_to(fmap)
    # Группы слоёв: города и потоки
    flows_anchor_layer = folium.FeatureGroup(
        name='Потоки в/из опорных пунктов', show=True)
    flows_other_layer = folium.FeatureGroup(name='Другие потоки', show=False)
    anchors_layer = folium.FeatureGroup(name='Опорные пункты', show=True)
    regular_layer = folium.FeatureGroup(name='Обычные города', show=True)
    potential_layer = folium.FeatureGroup(
        name='Потенциальные опорные пункты', show=True)
    polygons_layer = folium.FeatureGroup(name='Полигоны', show=True)

    # Слои
    edges_layer = folium.FeatureGroup(name='Потоки ≥ 1', show=True)
    edges_below_one = folium.FeatureGroup(name='Потоки < 1', show=False)
    cities_with_edges = folium.FeatureGroup(
        name='Города с потоками', show=True)
    cities_without_edges = folium.FeatureGroup(
        name='Города без потоков', show=False)
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
    scalar_map = ScalarMappable(norm=Normalize(
        vmin=log_min, vmax=log_max), cmap=cmap)

    def get_color(value):
        if value <= 0:
            return "#808080"
        log_val = np.log10(value + 1e-10)
        rgba = scalar_map.to_rgba(log_val)
        return "#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )

    # Tooltip
    self_sufficiency_df = analysis_results['self_sufficiency'].set_index(
        'city_id')

    # ---- Enhanced per-group tooltip data (override) ----
    # Prepare per-group self-sufficiency (0..1) and inflow per city
    per_group_self = None
    per_group_inflow = None
    group_names_order = []

    if isinstance(group_mobility, dict) and len(group_mobility) > 0:
        per_group_self = {}
        for grp, mob in group_mobility.items():
            df_self = mob.get('self_sufficiency', pd.DataFrame())
            if df_self is None or df_self.empty:
                continue
            s = df_self.set_index('city_id')['self_sufficiency_pct'] / 100.0
            per_group_self[grp] = s
            group_names_order.append(grp)

    if isinstance(group_matrices, dict) and len(group_matrices) > 0:
        per_group_inflow = {}
        for grp, mat in group_matrices.items():
            try:
                csc = csr_matrix(mat).tocsc()
                inflow = np.asarray(csc.sum(axis=0)).ravel()
                per_group_inflow[grp] = inflow
                if grp not in group_names_order:
                    group_names_order.append(grp)
            except Exception:
                continue

    potential_ids = set()
    pot_df0 = analysis_results.get('potential_anchors', pd.DataFrame())
    if isinstance(pot_df0, pd.DataFrame) and not pot_df0.empty and 'city_id' in pot_df0.columns:
        potential_ids = set(pot_df0['city_id'].tolist())

    def _fmt_self(val: float) -> str:
        return f"{val*100:.0f}%"

    def _block_header(text: str) -> str:
        return f"<div style=\"margin-top:6px; margin-bottom:2px; font-weight:600;\">{text}</div>"

    def _list_kv_aligned(items: list[tuple[str, str]]) -> str:
        if not items:
            return "<i>Нет данных</i>"
        rows = []
        for k, v in items:
            rows.append(
                "<div style=\"display:flex; align-items:baseline; gap:8px;\">"
                f"<span style=\"flex:1 1 auto; text-align:left;\">• {k}:</span>"
                f"<span style=\"flex:0 0 auto; text-align:right;\">{v}</span>"
                "</div>"
            )
        return "".join(rows)
    def _list_kv(items: list[tuple[str, str]]) -> str:
        if not items:
            return "<i>нет данных</i>"
        return "".join([f"<div>• {k}: {v}</div>" for k, v in items])

    # Override tooltip to include statuses and per-infrastructure details
    def create_tooltip(city_id):
        name = id_to_name[city_id]
        row = self_sufficiency_df.loc[city_id]

        is_anchor_city = bool(is_anchor.get(city_id, False))
        is_potential = (not is_anchor_city) and (city_id in potential_ids)

        html = [f"<b>{name}</b>"]
        # Population line right after city name
        try:
            pop_val = float(row.get("population", np.nan))
        except Exception:
            pop_val = np.nan
        if not np.isnan(pop_val):
            html.append(f"<div>Население, чел: <b>{int(round(pop_val))}</b></div>")
        if is_anchor_city:
            html.append("<div>Статус: <b>Опорный пункт</b></div>")
        elif is_potential:
            html.append(
                "<div>Статус: <b>Потенциальный опорный пункт</b></div>")

        # Градообслуживающие функции: самообеспеченность 0..1 по каждому типу
        if per_group_self:
            items = []
            for grp in group_names_order:
                s = per_group_self.get(grp)
                if s is None or city_id not in s.index:
                    continue
                share = float(s.loc[city_id])
                if not np.isnan(pop_val):
                    served = int(round(pop_val * share))
                    items.append((grp, f"{served:.0f} ({_fmt_self(share)})"))
                else:
                    items.append((grp, _fmt_self(share)))
            header_txt = "Градообслуживающие функции:" if (
                is_anchor_city or is_potential) else "Самообеспеченность:"
            html.append(_block_header(header_txt))
            html.append(_list_kv_aligned(items))
        else:
            header_txt = "Градообслуживающие функции:" if (
                is_anchor_city or is_potential) else "Самообеспеченность:"
            html.append(_block_header(header_txt))
            html.append("<i>нет данных по типам</i>")

        # Для обычного города ограничиваемся самообеспеченностью
        if not (is_anchor_city or is_potential):
            return "".join(html)

        # Градообразующие функции: приток людей по типам
        top_group = None
        top_value = -1.0
        if per_group_inflow:
            items = []
            total_inflow = 0.0
            for grp in group_names_order:
                infl = per_group_inflow.get(grp)
                if infl is None or city_id >= len(infl):
                    continue
                total_inflow += float(infl[city_id])
            for grp in group_names_order:
                infl = per_group_inflow.get(grp)
                if infl is None or city_id >= len(infl):
                    continue
                val = float(infl[city_id])
                if total_inflow > 0:
                    pct = (val / total_inflow) * 100.0
                    items.append((grp, f"{val:.0f} ({pct:.0f}%)"))
                else:
                    items.append((grp, f"{val:.0f}"))
                if val > top_value:
                    top_value = val
                    top_group = grp
            html.append(_block_header("Градообразующие функции:"))
            html.append(_list_kv_aligned(items))
        else:
            html.append(_block_header("Градообразующие функции:"))
            html.append("<i>нет данных по типам</i>")

        if top_group is not None:
            html.append(_block_header("Главная градообразующая функция:"))
            html.append(f"<div>{top_group}</div>")

        return "".join(html)

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

        # К/из опорных пунктов — отдельный слой
        if bool(is_anchor.get(from_id, False)) or bool(is_anchor.get(to_id, False)):
            line.add_to(flows_anchor_layer)
        else:
            line.add_to(flows_other_layer)

    # Отрисовка городов
    for city_id in gdf.index:
        point = id_to_geom[city_id]
        is_anchor_city = bool(is_anchor.get(city_id, False))
        # потенциальный опорный пункт, если не опорный и попадает в список potential_ids
        potential_ids = set()
        pot_df0 = analysis_results.get('potential_anchors', pd.DataFrame())
        if isinstance(pot_df0, pd.DataFrame) and not pot_df0.empty and 'city_id' in pot_df0.columns:
            potential_ids = set(pot_df0['city_id'].tolist())
        is_potential_city = (not is_anchor_city) and (city_id in potential_ids)

        color = 'red' if is_anchor_city else (
            '#00BFFF' if is_potential_city else 'green')
        radius = 4 if is_anchor_city else (3 if is_potential_city else 2)

        tip_html = create_tooltip(city_id)
        pop = folium.Popup(
            html=f"<div style='min-width:400px'>{tip_html}</div>", max_width=700)
        marker = folium.CircleMarker(
            location=(point.y, point.x),
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=tip_html,
            popup=pop,
        )

        if is_anchor_city:
            marker.add_to(anchors_layer)
        elif is_potential_city:
            marker.add_to(potential_layer)
        else:
            marker.add_to(regular_layer)

    # Добавляем слои

    polygons_layer.add_to(fmap)
    flows_anchor_layer.add_to(fmap)
    flows_other_layer.add_to(fmap)
    anchors_layer.add_to(fmap)
    potential_layer.add_to(fmap)
    regular_layer.add_to(fmap)

    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 260px; height: auto; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;">
        <b>Условные обозначения:</b><br>
        <i class="fa fa-circle" style="color:red"></i> Опорные пункты<br>
        <i class="fa fa-circle" style="color:#00BFFF"></i> Потенциальные опорные пункты<br>
        <i class="fa fa-circle" style="color:green"></i> Обычные города<br>
        <div style="margin-top:6px"><b>Интенсивность потоков</b></div>
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


def save_static_anchor_flow_png(
    matrix: csr_matrix,
    gdf: gpd.GeoDataFrame,
    analysis_results: dict,
    out_path: str,
    polygons_gdf: gpd.GeoDataFrame | None = None,
    *,
    focus: str = "anchors",
    max_edges: int = 5000,
    dpi: int = 200,
):
    """
    Render a static PNG map with anchor-related flows and cities using Matplotlib.

    - focus: "anchors" to show flows to/from anchors only, or "all" for all flows
    - max_edges: limit number of strongest flows drawn (by weight)
    Returns the saved path.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS; unable to plot.")

    # Work in WGS84 for simplicity
    gdf4326 = gdf.to_crs(epsg=4326) if gdf.crs.to_string().lower() != "epsg:4326" else gdf
    poly4326 = None
    if polygons_gdf is not None:
        poly4326 = polygons_gdf.to_crs(epsg=4326) if polygons_gdf.crs and polygons_gdf.crs.to_string().lower() != "epsg:4326" else polygons_gdf

    # Ensure city_id index
    if 'city_id' not in gdf4326.columns:
        gdf4326 = gdf4326.reset_index(drop=True)
        gdf4326['city_id'] = gdf4326.index
        gdf4326 = gdf4326.set_index('city_id')

    is_anchor = gdf4326['is_anchor_settlement'].astype(bool)
    id_to_point = gdf4326['geometry']

    # Extract edges
    rows, cols, data = find(matrix)
    if len(data) == 0:
        raise ValueError("Movement matrix is empty; nothing to plot.")

    # Filter to anchor-related flows if requested
    if focus == "anchors":
        mask = is_anchor.reindex(rows, fill_value=False).values | is_anchor.reindex(cols, fill_value=False).values
    else:
        mask = np.ones_like(data, dtype=bool)

    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]

    # Keep strongest flows
    if max_edges is not None and len(data) > max_edges:
        order = np.argsort(data)[-max_edges:]
        rows, cols, data = rows[order], cols[order], data[order]

    # Build line segments and colors
    nonzero = data[data > 0]
    vmin = np.log10(nonzero.min() + 1e-10)
    vmax = np.log10(nonzero.max() + 1e-10)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdYlGn_r')

    segments = []
    colors = []
    for r, c, w in zip(rows, cols, data):
        p_from = id_to_point.get(r)
        p_to = id_to_point.get(c)
        if p_from is None or p_to is None:
            continue
        segments.append([(p_from.x, p_from.y), (p_to.x, p_to.y)])
        colors.append(cmap(norm(np.log10(w + 1e-10))))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=dpi)

    # Basemap polygons
    if poly4326 is not None and not poly4326.empty:
        poly4326.plot(ax=ax, facecolor="#cddff7", edgecolor="#7a869a", linewidth=0.5, alpha=0.5)

    # Draw flows
    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=0.6, alpha=0.6)
        ax.add_collection(lc)

    # Plot cities
    anchors = gdf4326[is_anchor]
    regular = gdf4326[~is_anchor]
    if not regular.empty:
        regular.plot(ax=ax, markersize=6, color="#555555", alpha=0.8)
    if not anchors.empty:
        anchors.plot(ax=ax, markersize=20, color="#d62728", alpha=0.9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Anchor-related mobility flows")
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.02)
    plt.tight_layout()

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
