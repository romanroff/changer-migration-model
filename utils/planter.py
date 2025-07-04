import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import pickle

from sklearn.preprocessing import MinMaxScaler
from skmob.models.gravity import Gravity
from shapely.geometry import LineString
import warnings
import shapely
import skmob

from ._constant import *

def define_model():
    gravity = Gravity(
        deterrence_func_type="power_law",
        gravity_type="globally constrained",
        destination_exp=DESTINATION_EXP_COEFF,
        origin_exp=ORIGIN_EXP_COEFF,
        deterrence_func_args=[DETERRENCE_FUNC_COEFF],
    )
    return gravity

def drop_cities_no_population(df):
    assert "population" in df.columns, "population is not in df.columns"
    mask_not_empty_population_col = df["population"] > 0
    return df.loc[mask_not_empty_population_col, :]

def normalize_outflow_by_pop_mil(df):
    POPULATION_NORMALIZATION_VALUE = 1e6
    assert (
        "migrations_from_each_city" in df.columns and "population" in df.columns
    ), "migrations_from_each_city or population are not in df.columns"
    return cities["migrations_from_each_city"] * (
        cities["population"] / POPULATION_NORMALIZATION_VALUE
    )

def define_scaler():
    return MinMaxScaler()

def scale_cities_attrs(df, cols_to_scale, scaler, fit=True):
    """
    скейлить думаю тоже можно один раз
    потом сохранить скейлер и только для измененных параметров использовать
    """
    if fit:
        return scaler.fit_transform(df.loc[:,cols_to_scale])
    else:
        df = df.T
        return scaler.transform(df.loc[:,cols_to_scale,])

def calculate_attractiveness(df):
    """
    это как будто тоже один раз и потом просто пересчитывать для одного города

    Почему вес/привлекательность так сделано? Взвешиваются отдельно фабрики-зарплаты, качества города и климат.
    + Я попробовал по-разному их взвешивать (и задавать различные значения для гравити-модели)
    и такая комюинация показала себя лучше всего. Можно считать это экспертной оценкой.
    """

    assert all(
        param in df.columns
        for param in [
            "factories_total",
            "median_salary",
            "ueqi_residential",
            "ueqi_green_spaces",
            "ueqi_public_and_business_infrastructure",
            "ueqi_social_and_leisure_infrastructure",
            "ueqi_citywide_space",
            "harsh_climate",
        ]
    )

    city_attractiveness_coeff = round(
    (
        FACTORY_SALARY_W_COEFF * (df["factories_total"] + df["median_salary"]) * CITY_PARAMS_W_COEFF
        + (
            df["ueqi_residential"]
            + df["ueqi_street_networks"]
            + df["ueqi_green_spaces"]
            + df["ueqi_public_and_business_infrastructure"]
            + df["ueqi_social_and_leisure_infrastructure"]
            + df["ueqi_citywide_space"]
        )
        + (1 - df["harsh_climate"])
        + 1
    ).item(),
    0,)
    return city_attractiveness_coeff

def filter_od_matrix_resetted(df):
    """
    вот это кстати можно и один раз сделать
    """
    assert all(param in df.columns for param in ["origin", "destination"])
    mask_od_origin_not_destination = df["origin"] != df["destination"]
    return df.loc[mask_od_origin_not_destination, :]

def reset_od_matrix(od_matrix):
    # Reset index to have 'origin' as a column
    od_matrix_reset = od_matrix.reset_index()
    od_matrix_reset.rename(columns={"region_city": "origin"}, inplace=True)
    od_matrix_reset = pd.melt(
        od_matrix_reset, id_vars=["origin"], var_name="destination", value_name="flow"
    )
    return od_matrix_reset

def check_filter_cities_in_od_matrix(df, od_df):
    assert "region_city" in df and all(
        param in od_df for param in ["origin", "destination"]
    )
    mask_cities_in_od = df["region_city"].isin(
        set(od_df["origin"].to_list() + od_df["destination"].to_list())
    )
    return df.loc[mask_cities_in_od, :]

def make_flow_df(od_df, df_with_od_geoms):
    fdf = skmob.FlowDataFrame(
        data=od_df,
        origin="origin",
        destination="destination",
        flow="flow",
        tessellation=df_with_od_geoms,
        tile_id="region_city",
    )
    return fdf

def fit_flow_df(fdf, gravity) -> None:
    gravity.fit(fdf, relevance_column="city_attractiveness_coeff")

def generate_flows(df, gravity):
    """
    генерирует таблицу с OD и потоком между
    !!! нужно пересчитыват какждый раз при изменении параметров города
    """
    assert all(
        param in df.columns
        for param in ["city_attractiveness_coeff", "region_city", "norm_outflow"]
    )

    fdf_fitted = gravity.generate(
        df,
        relevance_column="city_attractiveness_coeff",
        tot_outflows_column="norm_outflow",
        out_format="flows",
        tile_id_column="region_city",
    )
    return pd.DataFrame(fdf_fitted).sort_values(by=["flow", "destination"])

def inverse_scale_df(df, cols, scaler):
    return pd.DataFrame(scaler.inverse_transform(df.loc[:, cols]), columns=cols)

def tailor_geometries(gdf):
    GRID_SIZE = 0.001
    return shapely.set_precision(gdf["geometry"].array, grid_size=GRID_SIZE)

def post_processing(gdf: pd.DataFrame):
    assert all(attr in gdf.columns for attr in ["geometry", "flow"])
    MINIMAL_FLOW = 1e-3  # anything beyond is a noise
    MAXIMAL_FLOW = 4  # anything beyond is a noise
    gdf["distance"] = (
        gpd.GeoSeries(gdf["geometry"], crs=DEGREE_CRS).to_crs(METRIC_CRS).length
    )
    mask6 = (
        (gdf["distance"] < DISTANCE_TRASHOLD_METERS)
        & (gdf["flow"] > MINIMAL_FLOW)
        & (gdf["flow"] < MAXIMAL_FLOW)
    )
    gdf.loc[:, "big_flows"] = (np.log(gdf.loc[:, "flow"] + 5) ** 10).astype(
        int
    )  # some empirical constants
    gdf.loc[:, "scaled_flows_forvis"] = round(
        np.log(gdf.loc[:, "big_flows"]) / 1e2, 3
    )  # some empirical constants
    return gdf, mask6

def join_od_params(fdf_with_flows, cities):
    df_links = fdf_with_flows.merge(
        cities[["region_city", "city_attractiveness_coeff", "population"]].rename(
            columns={
                "region_city": "destination",
                "city_attractiveness_coeff": "destination_attr",
            }
        ),
        left_on="destination",
        right_on="destination",
    ).merge(
        cities[["region_city", "city_attractiveness_coeff"]].rename(
            columns={
                "city_attractiveness_coeff": "origin_attr",
                "region_city": "origin",
            }
        ),
        left_on="origin",
        right_on="origin",
    )
    return df_links

# Define the function that uses the pre-constructed dictionary
def create_linestring(row, geometry_dict):
    origin = row["origin"]
    destination = row["destination"]
    # Check if both origin and destination exist in the dictionary
    if origin in geometry_dict and destination in geometry_dict:
        return LineString([geometry_dict[origin], geometry_dict[destination]])
    return None

def make_od_linestring_geom(fdf_fitted_df, init_cities):
    """
    Геометрии ставятся один раз --- БРАТЬ ИЗ ФАЙЛА
    """
    # Create a dictionary that maps region cities to their geometries for faster access
    city_geometry_dict = init_cities.set_index("region_city")["geometry"].to_dict()
    # Apply the function in parallel
    return fdf_fitted_df.parallel_apply(
        lambda row: create_linestring(row, city_geometry_dict), axis=1
    )

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

    def __getitem__(self, key):
        return getattr(self, key, f"Property '{key}' not found")

    def __setitem__(self, key, value):
        if hasattr(self, key):
            print("Warning: rewriting existing attribute")
        setattr(self, key, value)

    def save_initial_state(self):
        """Save the initial state of cities dataframe."""
        if self.initial_cities_state is None:
            self.initial_cities_state = self.cities.copy()
            self.initial_links_state = self.gdf_links.copy()
            print("Initial cities state saved.")

    def save_previous_state(self):
        """Save the previous state of cities dataframe."""
        self.prev_cities_state = self.cities.copy()
        self.prev_links_state = self.gdf_links.copy()

    def save_current_state(self):
        """Save the previous state of cities dataframe."""
        self.current_cities_state = self.cities.copy()
        self.current_links_state = self.gdf_links.copy()

    def compare_city_states(self):
        """Compare two states of the cities DataFrame or any other DataFrame."""
        # You can compare the full dataframe or specific columns
        if hasattr(self, "current_cities_state") and hasattr(
            self, "initial_cities_state"
        ):
            diff_cities = self.current_cities_state[
                ["flows_in", "flows_out", "region_city", "geometry"]
            ].merge(
                self.initial_cities_state[
                    ["flows_in", "flows_out", "region_city"]
                ].rename(
                    columns={"flows_in": "flows_in_prev", "flows_out": "flows_out_prev"}
                )
            )

            diff_cities["in_diff"] = (
                diff_cities["flows_in"] - diff_cities["flows_in_prev"]
            )

            diff_cities["out_diff"] = (
                diff_cities["flows_out"] - diff_cities["flows_out_prev"]
            )

            diff_cities["in_out_diff"] = (
                diff_cities["in_diff"] - diff_cities["out_diff"]
            )

            # Set a threshold for filtering small fluctuations
            threshold = 3  # Adjust this value based on your data
            # Filter out points with 'in_out_diff' below the threshold
            mask_fluctuation = diff_cities["in_out_diff"].abs() <= threshold
            diff_cities.loc[mask_fluctuation, "in_out_diff"] = 0

            return diff_cities[
                ["region_city", "geometry", "in_out_diff", "in_diff", "out_diff"]
            ].to_crs(DEGREE_CRS)
        else:
            print("Both states must be DataFrame objects.")
            return False

    def compare_link_states(self):
        """Compare two states of the cities DataFrame or any other DataFrame."""
        # You can compare the full dataframe or specific columns
        if hasattr(self, "current_cities_state") and hasattr(
            self, "initial_links_state"
        ):
            diff_links = (
                self.initial_links_state[
                    [
                        "origin",
                        "destination",
                        "big_flows",
                        "geometry",
                        "scaled_flows_forvis",
                    ]
                ]
                .rename(columns={"big_flows": "init_flows"})
                .merge(
                    self.current_links_state[["origin", "destination", "big_flows"]],
                )
            )

            diff_links["big_flows"] = diff_links["big_flows"] - diff_links["init_flows"]

            # Set a threshold for filtering small fluctuations
            threshold = 3  # Adjust this value based on your data
            # Filter out points with 'in_out_diff' below the threshold
            mask_fluctuation = diff_links["big_flows"].abs() <= threshold
            diff_links.loc[mask_fluctuation, "big_flows"] = 0

            return diff_links.drop(columns=["init_flows"])
        else:
            print("Both states must be DataFrame objects.")
            return False

    def reset_state(self):
        """Reset to the initial state of cities."""
        if self.initial_cities_state is not None:
            self.cities = self.initial_cities_state.copy()
            print("Cities state reset to the initial state.")
        else:
            print("No initial state to reset to.")

    @classmethod
    def make_scaler(cls):
        cls.scaler = define_scaler()

    @classmethod
    def make_model(cls):
        cls.model = define_model()

    def mark_stage_dirty(self, stage_number):
        # Mark a stage and all subsequent stages as needing rerun
        for stage in range(stage_number, max(self.pipeline_stages.keys()) + 1):
            self.pipeline_stages[stage] = False

    def run_cities_pipeline_stage_1(self):
        if not self.pipeline_stages[1]:
            if hasattr(self, "cities"):
                self.cities = drop_cities_no_population(self.cities)
                self.cities["norm_outflow"] = normalize_outflow_by_pop_mil(self.cities)
                self.pipeline_stages[1] = True
                self.mark_stage_dirty(2)  # Mark later stages as needing rerun
            else:
                warnings.warn("Please provide 'cities' data")
        else:
            print("Skipping: Stage 1 has already been run")

    def run_cities_pipeline_stage_2(self):
        if not self.pipeline_stages[2]:
            if hasattr(self, "cities"):
                self.init_cities = self.cities.copy().to_crs(DEGREE_CRS)
                self.init_cities["geometry"] = tailor_geometries(self.init_cities)
                self.pipeline_stages[2] = True
                self.mark_stage_dirty(3)  # Stage 3 depends on Stage 2
            else:
                warnings.warn("Please provide 'cities' data")
        else:
            print("Skipping: Stage 2 has already been run")

    def run_cities_pipeline_stage_3(self):
        if not self.pipeline_stages[3]:
            if hasattr(self, "cities") and hasattr(self, "od"):
                self.od_matrix_reset = reset_od_matrix(self["od"])
                self.od_matrix_reset = filter_od_matrix_resetted(self.od_matrix_reset)
                self.od_matrix_reset.reset_index(drop=True, inplace=True)
                # self.cities = check_filter_cities_in_od_matrix(
                #     self.cities, self.od_matrix_reset
                # )
                self.pipeline_stages[3] = True
                self.mark_stage_dirty(4)  # Stage 4 depends on Stage 3
            else:
                warnings.warn("Please provide 'cities' and 'od' data")
        else:
            print("Skipping: Stage 3 has already been run")

    def run_cities_pipeline_stage_4(self):
        if not self.pipeline_stages[4]:
            if hasattr(self, "cities") and hasattr(self, "scaler"):

                self.cities.loc[:, self.cols] = scale_cities_attrs(
                    self.cities, self.cols, self.scaler, fit=True
                )
                self.scaled_cities = self.cities.copy()
                self.cities["city_attractiveness_coeff"] = calculate_attractiveness(
                    self.cities
                )
                self.init_cities["city_attractiveness_coeff"] = self.cities[
                    "city_attractiveness_coeff"
                ].copy()

                self.pipeline_stages[4] = True
                self.mark_stage_dirty(5)  # Stage 5 depends on Stage 4
            else:
                warnings.warn("Please provide 'cities' data and a scaler")
        else:
            print("Skipping: Stage 4 has already been run")

    def run_cities_pipeline_stage_5(self):
        if not self.pipeline_stages[5]:
            if hasattr(self, "cities") and hasattr(self, "od_matrix_reset"):

                self.fdf = make_flow_df(self.od_matrix_reset, self.cities)
                self.cities.loc[:, self.cols] = inverse_scale_df(
                    self.cities, self.cols, self.scaler
                )
                self.cities.loc[:, self.cols_to_round] = self.cities.loc[
                    :, self.cols_to_round
                ].astype(int)
                self.pipeline_stages[5] = True
                self.mark_stage_dirty(6)  # Stage 6 depends on Stage 5
            else:
                warnings.warn("Please provide 'cities' and 'od_matrix_reset' data")
        else:
            print("Skipping: Stage 5 has already been run")

    def run_cities_pipeline_stage_6(self):
        if not self.pipeline_stages[6]:
            if hasattr(self, "fdf") and hasattr(self, "init_cities"):
                fit_flow_df(self.fdf, self.model)  # Fit the model with flow data
                self.fdf_fitted_df = generate_flows(self.cities, self.model)
                self.pipeline_stages[6] = True
                self.mark_stage_dirty(7)  # Stage 7 depends on Stage 6
            else:
                warnings.warn("Please provide 'fdf' and 'init_cities' data")
        else:
            print("Skipping: Stage 6 has already been run")

    def run_cities_pipeline_stage_7(self):
        if not self.pipeline_stages[7]:
            if hasattr(self, "fdf_fitted_df") and hasattr(self, "init_cities"):
                self.od_linestrings = make_od_linestring_geom(
                    self.fdf_fitted_df, self.init_cities
                )
                self.fdf_fitted_df["geometry"] = self.od_linestrings
                self.pipeline_stages[7] = True
                self.mark_stage_dirty(8)  # Stage 8 depends on Stage 7
            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'init_cities' data")
        else:
            print("Skipping: Stage 7 has already been run")

    def run_cities_pipeline_stage_8(self):
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
                    (
                        self.gdf_links.drop(columns=["destination", "geometry"])
                        .groupby("origin")
                        .sum()
                        .reset_index(drop=False)
                    )
                    .loc[:, ["origin", "big_flows"]]
                    .rename(columns={"big_flows": "flows_out", "origin": "region_city"})
                )

                flows_grouped_in = (
                    (
                        self.gdf_links.drop(columns=["origin", "geometry"])
                        .groupby("destination")
                        .sum()
                        .reset_index(drop=False)
                    )
                    .loc[:, ["destination", "big_flows"]]
                    .rename(
                        columns={"big_flows": "flows_in", "destination": "region_city"}
                    )
                )

                self.cities = self.cities.merge(flows_grouped_in, how="left").merge(
                    flows_grouped_out, how="left"
                )

                self.save_initial_state()
                self.save_current_state()

            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'cities' data")
        else:
            print("Skipping: Stage 8 has already been run")

    # -----------------------------------------------------------------
    def run_cities_pipeline_stage_4_upd(self):
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

                    self.cities.loc[
                        self.update_city_name_idx, "city_attractiveness_coeff"
                    ] = calculate_attractiveness(
                        self.scaled_cities.loc[self.update_city_name_idx, :]
                        .to_frame()
                        .T
                    ).item()

                    self.pipeline_stages[4] = True
                    self.mark_stage_dirty(5)  # Stage 5 depends on Stage 4
            else:
                warnings.warn("Please provide 'cities' data and a scaler")
        else:
            print("Skipping: Stage 4 has already been run")

    def run_cities_pipeline_stage_5_upd(self):
        if not self.pipeline_stages[5]:
            if hasattr(self, "cities") and hasattr(self, "od_matrix_reset"):

                self.cities.loc[self.update_city_name_idx, self.cols_to_round] = (
                    self.cities.loc[
                        self.update_city_name_idx, self.cols_to_round
                    ].astype(int)
                )

                self.pipeline_stages[5] = True
                self.mark_stage_dirty(6)  # Stage 6 depends on Stage 5
            else:
                warnings.warn("Please provide 'cities' and 'od_matrix_reset' data")
        else:
            print("Skipping: Stage 5 has already been run")

    def run_cities_pipeline_stage_6_upd(self):
        if not self.pipeline_stages[6]:
            if hasattr(self, "fdf") and hasattr(self, "init_cities"):

                # fit_flow_df(self.fdf, self.model)  # Fit the model with flow data
                self.fdf_fitted_df = generate_flows(self.cities, self.model)

                self.pipeline_stages[6] = True
                self.mark_stage_dirty(7)  # Stage 7 depends on Stage 6
            else:
                warnings.warn("Please provide 'fdf' and 'init_cities' data")
        else:
            print("Skipping: Stage 6 has already been run")

    def run_cities_pipeline_stage_7_upd(self):
        if not self.pipeline_stages[7]:
            fitted_df = getattr(self, "fdf_fitted_df", None)
            if fitted_df is not None:
                # Use direct assignment to avoid unnecessary copies
                fitted_df["geometry"] = self.od_linestrings
                self.pipeline_stages[7] = True
                self.mark_stage_dirty(8)  # Marking stage 8 as dependent

            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'init_cities' data")
        else:
            print("Skipping: Stage 7 has already been run")

    def run_cities_pipeline_stage_8_upd(self):
        if not self.pipeline_stages[8]:
            fitted_df = getattr(self, "fdf_fitted_df", None)
            cities_df = getattr(self, "cities", None)

            if fitted_df is not None and cities_df is not None:
                self.df_links = join_od_params(fitted_df, cities_df)
                # Perform post-processing and get masks in a single step if possible
                self.df_links, self.mask_distance_flow = post_processing(self.df_links)

                # Create GeoDataFrame directly without reassignment
                self.gdf_links = gpd.GeoDataFrame(
                    self.df_links.loc[self.mask_distance_flow], crs=DEGREE_CRS
                )
                self.gdf_links["geometry"] = tailor_geometries(self.gdf_links)

                self.pipeline_stages[8] = True

                # Optimize groupby operations by using fewer temporary DataFrames
                flows_out = (
                    self.gdf_links.drop(columns=["destination", "geometry"])
                    .groupby("origin", as_index=False)["big_flows"]
                    .sum()
                    .rename(columns={"big_flows": "flows_out", "origin": "region_city"})
                )

                # Drop columns in place to minimize data copying
                if "flows_in" in cities_df.columns:
                    cities_df.drop(columns=["flows_in", "flows_out"], inplace=True)

                flows_in = (
                    self.gdf_links.drop(columns=["origin", "geometry"])
                    .groupby("destination", as_index=False)["big_flows"]
                    .sum()
                    .rename(
                        columns={"big_flows": "flows_in", "destination": "region_city"}
                    )
                )

                # Merge flows more efficiently
                self.cities = cities_df.merge(flows_in, how="left").merge(
                    flows_out, how="left"
                )

                self.save_current_state()

            else:
                warnings.warn("Please provide 'fdf_fitted_df' and 'cities' data")
        else:
            print("Skipping: Stage 8 has already been run")

    # -----------------------------------------------------------------

    def update_city_params(self, city_name, new_params):
        # Check if the city exists in the DataFrame

        self.save_previous_state()

        self.update_city_name = city_name
        self.updated_city_params = new_params

        if city_name in self.cities["region_city"].values:
            # Update the DataFrame for the specific city
            self.update_city_name_idx = self.cities[
                self.cities["region_city"] == city_name
            ].index.item()

            self.cities.loc[self.update_city_name_idx, new_params.keys()] = (
                new_params.values()
            )
            print(f"Updated parameters for {city_name}")
            # Mark relevant stages as dirty
            self.mark_stage_dirty(
                4
            )  # Stage 4 needs rerunning after updating city params
        else:
            print(f"City {city_name} not found in the DataFrame.")

    def recalculate_after_update(self):
        """
        Updates city parameters and recalculates the pipeline from Stage 4 to Stage 8.

        :param city_name: Name of the city whose parameters need to be updated
        :param new_params: Dictionary of the new parameters to update the city with
        """

        # Step 2: Re-run necessary pipeline stages after update
        print(f"Recalculating after updating parameters")

        self.run_cities_pipeline_stage_4_upd()  # Recalculate Stage 4
        self.run_cities_pipeline_stage_5_upd()  # Recalculate Stage 5
        self.run_cities_pipeline_stage_6_upd()  # Recalculate Stage 6
        self.run_cities_pipeline_stage_7_upd()  # Recalculate Stage 7
        self.run_cities_pipeline_stage_8_upd()  # Recalculate Stage 8

        print(f"Recalculation complete.")

    def to_pickle(self, filename):
        """Save the whole class instance to a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Class instance saved to {filename}")

    @classmethod
    def from_pickle(cls, filename):
        """Load the class instance from a pickle file."""
        with open(filename, "rb") as f:
            instance = pickle.load(f)
        print(f"Class instance loaded from {filename}")
        return instance
