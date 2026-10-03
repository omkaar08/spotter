"""
Spotter ML Assessment - Feature Engineering & Preprocessing Pipeline

This module encapsulates data cleaning, missing value imputation, spatial features,
domain-specific freight physics, temporal decomposition, and target encodings.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculates great circle distance between coordinate pairs in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = c * 6367.0 * 0.621371
    return miles


def calculate_bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculates spatial bearing angle in degrees between pickup and delivery coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(x, y)
    bearing_deg = (np.degrees(initial_bearing) + 360) % 360
    return bearing_deg


class FreightFeaturePipeline:
    """
    Stateful feature pipeline that learns lookup statistics from training data
    and applies leakage-free transformations to train, validation, and December datasets.
    """

    def __init__(self):
        self.equipment_weight_medians: Dict[str, float] = {}
        self.global_weight_median: float = 32000.0
        self.global_market_index_median: float = 0.95
        
        # City coordinate lookups
        self.city_lat_map: Dict[str, float] = {}
        self.city_lon_map: Dict[str, float] = {}
        
        # Target encoding lookups (Rate Per Mile)
        self.lane_rpm_map: Dict[str, float] = {}
        self.pickup_rpm_map: Dict[str, float] = {}
        self.delivery_rpm_map: Dict[str, float] = {}
        self.equipment_rpm_map: Dict[str, float] = {}
        self.global_mean_rpm: float = 2.20

        # Daily market & quote signal lookups from validation/train data
        self.daily_market_map: Dict[str, float] = {}
        self.daily_quote_map: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> 'FreightFeaturePipeline':
        """Learns feature statistics and lookup maps from training data."""
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # Weight medians
        self.equipment_weight_medians = data.groupby('equipment')['weight'].median().to_dict()
        self.global_weight_median = float(data['weight'].median())
        self.global_market_index_median = float(data['market_index'].median())

        # City coordinates
        for city, group in data.groupby('pickup'):
            self.city_lat_map[city] = float(group['pickup_lat'].iloc[0])
            self.city_lon_map[city] = float(group['pickup_lon'].iloc[0])
            
        for city, group in data.groupby('delivery'):
            self.city_lat_map[city] = float(group['delivery_lat'].iloc[0])
            self.city_lon_map[city] = float(group['delivery_lon'].iloc[0])

        # Rate Per Mile target encoding
        data['rpm'] = data['posted_rate'] / data['distance']
        self.global_mean_rpm = float(data['rpm'].mean())

        data['lane'] = data['pickup'] + '_' + data['delivery']
        self.lane_rpm_map = data.groupby('lane')['rpm'].mean().to_dict()
        self.pickup_rpm_map = data.groupby('pickup')['rpm'].mean().to_dict()
        self.delivery_rpm_map = data.groupby('delivery')['rpm'].mean().to_dict()
        self.equipment_rpm_map = data.groupby('equipment')['rpm'].mean().to_dict()

        # Daily market index & quote signal lookups
        date_str = data['date'].dt.strftime('%Y-%m-%d')
        self.daily_market_map = data.groupby(date_str)['market_index'].median().to_dict()
        self.daily_quote_map = data.groupby(date_str)['quote_signal'].median().to_dict()

        return self

    def update_daily_lookups(self, val_df: pd.DataFrame) -> None:
        """Enriches daily market index and quote signal lookups from validation set for out-of-sample dates (e.g. Nov-Dec)."""
        data = val_df.copy()
        data['date'] = pd.to_datetime(data['date'])
        date_str = data['date'].dt.strftime('%Y-%m-%d')
        
        val_market = data.groupby(date_str)['market_index'].median().to_dict()
        val_quote = data.groupby(date_str)['quote_signal'].median().to_dict()
        
        self.daily_market_map.update(val_market)
        self.daily_quote_map.update(val_quote)

    def transform(self, df: pd.DataFrame, is_december_input: bool = False) -> pd.DataFrame:
        """Transforms raw load data into model-ready numerical feature matrix."""
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        date_str = data['date'].dt.strftime('%Y-%m-%d')

        # Fill missing coordinates for December inputs
        if 'pickup_lat' not in data.columns or data['pickup_lat'].isna().any():
            data['pickup_lat'] = data['pickup'].map(self.city_lat_map).fillna(37.0)
            data['pickup_lon'] = data['pickup'].map(self.city_lon_map).fillna(-85.0)
            data['delivery_lat'] = data['delivery'].map(self.city_lat_map).fillna(41.0)
            data['delivery_lon'] = data['delivery'].map(self.city_lon_map).fillna(-85.0)

        # Impute missing weight
        if 'weight' in data.columns:
            data['weight'] = data['weight'].fillna(data['equipment'].map(self.equipment_weight_medians))
            data['weight'] = data['weight'].fillna(self.global_weight_median)
        else:
            data['weight'] = self.global_weight_median

        # Impute missing market index
        if 'market_index' not in data.columns or is_december_input:
            data['market_index'] = date_str.map(self.daily_market_map).fillna(self.global_market_index_median)
        else:
            data['market_index'] = data['market_index'].fillna(date_str.map(self.daily_market_map))
            data['market_index'] = data['market_index'].fillna(self.global_market_index_median)

        # Impute missing quote signal
        if 'quote_signal' not in data.columns or is_december_input:
            data['quote_signal'] = date_str.map(self.daily_quote_map).fillna(2.05)
        else:
            data['quote_signal'] = data['quote_signal'].fillna(date_str.map(self.daily_quote_map))
            data['quote_signal'] = data['quote_signal'].fillna(2.05)

        # Spatial Features
        data['haversine_dist'] = haversine_distance(
            data['pickup_lat'].values, data['pickup_lon'].values,
            data['delivery_lat'].values, data['delivery_lon'].values
        )
        data['circuitousness'] = data['distance'] / (data['haversine_dist'] + 1e-5)
        data['bearing'] = calculate_bearing(
            data['pickup_lat'].values, data['pickup_lon'].values,
            data['delivery_lat'].values, data['delivery_lon'].values
        )
        data['mid_lat'] = (data['pickup_lat'] + data['delivery_lat']) / 2.0
        data['mid_lon'] = (data['pickup_lon'] + data['delivery_lon']) / 2.0

        # Domain Freight Physics
        data['quote_rate'] = data['distance'] * data['quote_signal']
        data['market_quote_rate'] = data['distance'] * data['quote_signal'] * data['market_index']
        data['weight_per_mile'] = data['weight'] / (data['distance'] + 1.0)
        data['weight_ton'] = data['weight'] / 2000.0

        # Temporal Features
        data['month'] = data['date'].dt.month
        data['dayofweek'] = data['date'].dt.dayofweek
        data['dayofyear'] = data['date'].dt.dayofyear
        data['quarter'] = data['date'].dt.quarter
        data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
        
        # Cyclical Temporal Encodings
        data['sin_dayofweek'] = np.sin(2 * np.pi * data['dayofweek'] / 7.0)
        data['cos_dayofweek'] = np.cos(2 * np.pi * data['dayofweek'] / 7.0)
        data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12.0)
        data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12.0)
        data['sin_dayofyear'] = np.sin(2 * np.pi * data['dayofyear'] / 365.25)
        data['cos_dayofyear'] = np.cos(2 * np.pi * data['dayofyear'] / 365.25)

        # Categorical Equipment Flags
        for eq in ['Dry Van', 'Flatbed', 'Reefer']:
            data[f'equipment_{eq}'] = (data['equipment'] == eq).astype(int)

        # Target Encoding Maps
        data['lane'] = data['pickup'] + '_' + data['delivery']
        data['lane_mean_rpm'] = data['lane'].map(self.lane_rpm_map).fillna(self.global_mean_rpm)
        data['pickup_mean_rpm'] = data['pickup'].map(self.pickup_rpm_map).fillna(self.global_mean_rpm)
        data['delivery_mean_rpm'] = data['delivery'].map(self.delivery_rpm_map).fillna(self.global_mean_rpm)
        data['equipment_mean_rpm'] = data['equipment'].map(self.equipment_rpm_map).fillna(self.global_mean_rpm)

        return data

    @property
    def feature_columns(self) -> List[str]:
        return [
            'distance', 'haversine_dist', 'circuitousness', 'bearing', 'mid_lat', 'mid_lon',
            'weight', 'weight_per_mile', 'weight_ton',
            'market_index', 'quote_signal', 'quote_rate', 'market_quote_rate',
            'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
            'month', 'dayofweek', 'dayofyear', 'quarter', 'is_weekend',
            'sin_dayofweek', 'cos_dayofweek', 'sin_month', 'cos_month',
            'sin_dayofyear', 'cos_dayofyear',
            'equipment_Dry Van', 'equipment_Flatbed', 'equipment_Reefer',
            'lane_mean_rpm', 'pickup_mean_rpm', 'delivery_mean_rpm', 'equipment_mean_rpm'
        ]
