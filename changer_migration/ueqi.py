import pandas as pd
import numpy as np

UEQI_GROUPS = {
    "ueqi_residential": [
        "capacity_delivery_point__post_office", "capacity_hairdresser__beauty",
        "capacity_domestic_services"
    ],
    "ueqi_street_networks": [
        "capacity_public_transport_stop", "capacity_parking", "capacity_gas_station",
        "capacity_railway_station", "capacity_car_service"
    ],
    "ueqi_green_spaces": [
        "capacity_playground", "capacity_square__boulevard__forest_park",
        "capacity_public_space", "capacity_park", "capacity_amusement_park", "capacity_dog_park"
    ],
    "ueqi_public_and_business_infrastructure": [
        "capacity_cafe__coffee", "capacity_bar_restaurant", "capacity_food_court",
        "capacity_convenience", "capacity_houseware", "capacity_supermarket", "capacity_zoo",
        "capacity_hypermarket", "capacity_specialized_store",
        "capacity_health_center__dispensary", "capacity_pharmacy", "capacity_polyclinic",
        "capacity_local_hospital", "capacity_city_hospital", "capacity_commercial_clinic",
        "capacity_bank"
    ],
    "ueqi_social_and_leisure_infrastructure": [
        "capacity_kindergarten", "capacity_school", "capacity_community_center", "capacity_art_school",
        "capacity_universal_hall", "capacity_community_centre_culture_house",
        "capacity_library", "capacity_cult_object",
        "capacity_workout__school_gym", "capacity_gym__fitness_center",
        "capacity_skatepark__workout_for_teenagers", "capacity_swimming_pool"
    ],
    "ueqi_citywide_space": [
        "capacity_local_police", "capacity_police_supporting_point", "capacity_fire_station"
    ]
}

def calculate_ueqi(df, groups):
    """
    Calculate Urban Environment Quality Index (UEQI) for different service groups.
    
    This function processes a DataFrame to calculate quality indices for various urban service
    groups by summing service capacities within each group, normalizing by population,
    and applying constraints.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing service capacity data and population.
                              Expected to have columns with 'capacity_' prefix for services
                              and a 'population' column.
        groups (dict): Dictionary where keys are group names (str) and values are lists
                      of service names (list of str). Service names should correspond
                      to column names in df when prefixed with 'capacity_'.
    
    Returns:
        pandas.DataFrame: Modified DataFrame with additional columns for each group,
                         where each group column contains the calculated UEQI values
                         (float, range 0-100, rounded to 3 decimal places).
    
    Note:
        - Only existing services (columns present in df) are included in calculations
        - Values are normalized per 100 people (capacity sum / (population / 100))
        - Final values are capped at maximum of 100
        - Results are rounded to 3 decimal places
    """
    for group_name, services in groups.items():
        # Проверяем, существуют ли сервисы в датафрейме
        existing_services = [s for s in services if s in df.columns]
        
        # Суммируем все сервисы в группе и делим на население (в процентах)
        df[group_name] = df[existing_services].sum(axis=1) / (df['population'] / 100)
        
        # Применяем ограничение max 100
        df[group_name] = np.minimum(100, round(df[group_name], 3))
    
    return df

# Function for counting of changing capacity
def count_change_capacity(df: pd.DataFrame, name: str, percent: float):
    """
    Calculate the change in capacity based on a given percentage of the population for a specific town.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing population and town_name columns.
    name (str): The name of the town for which to calculate the change in capacity.
    percent (float): The percentage of the population to use for calculating the change in capacity.

    Returns:
    float: The calculated change in capacity for the specified town.

    Notes:
    - The DataFrame `df` is expected to have at least two columns: 'population' and 'town_name'.
    - The 'population' column should contain numerical values representing the population of each town.
    - The 'town_name' column should contain string values representing the names of the towns.
    - The function calculates the capacity by multiplying the population of the specified town by the given percentage and then dividing by 100.
    - The result is returned as a float value.
    """
    
    capacity = df.population / 100 * percent
    return int(capacity.loc[df.town_name == name].item())

def get_ueqi(wff, name):
    city = wff.cities[wff.cities['region_city'].str.contains(name)]
    return city[
                ['region_city',
                "ueqi_residential",
                "ueqi_street_networks",
                "ueqi_green_spaces",
                "ueqi_public_and_business_infrastructure",
                "ueqi_social_and_leisure_infrastructure",
                "ueqi_citywide_space"]
            ].round(3)