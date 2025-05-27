#!/usr/bin/env python3
"""
F1 Fantasy Optimizer Suite
Complete pipeline for VFM calculation, track affinity analysis, and team optimization
"""

import pandas as pd
import numpy as np
import re
import sys
import os
import json
import itertools
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def get_races_completed(base_path):
    """Calculate number of races completed based on race columns in data file"""
    try:
        driver_data = pd.read_csv(f'{base_path}driver_race_data.csv')
        race_columns = [col for col in driver_data.columns if col.startswith('Race')]
        return len(race_columns)
    except Exception as e:
        print(f"Error reading driver data: {e}")
        return None


def get_user_configuration():
    """Get configuration from user input"""
    print("F1 Fantasy Optimizer Configuration")
    print("="*80)
    
    config = {}
    
    # Base path
    print("\nData Location:")
    default_path = '/content/drive/MyDrive/F1 Fantasy/'
    base_path = input(f"Enter base path for data files [{default_path}]: ").strip()
    config['base_path'] = base_path if base_path else default_path
    
    # Ensure path ends with /
    if not config['base_path'].endswith('/'):
        config['base_path'] += '/'
    
    # Calculate races completed
    races_completed = get_races_completed(config['base_path'])
    if races_completed is None:
        print("Error: Could not read driver data file.")
        sys.exit(1)
    config['races_completed'] = races_completed
    print(f"\nDetected {races_completed} races completed.")
    
    # Current team - Drivers
    print("\nCurrent Team - Drivers (5 required):")
    drivers = []
    for i in range(5):
        while True:
            driver = input(f"  Driver {i+1}: ").strip()
            if driver:
                drivers.append(driver)
                break
            print("  Driver name cannot be empty.")
    config['current_drivers'] = drivers
    
    # Current team - Constructors
    print("\nCurrent Team - Constructors (2 required):")
    constructors = []
    for i in range(2):
        while True:
            constructor = input(f"  Constructor {i+1}: ").strip()
            if constructor:
                constructors.append(constructor)
                break
            print("  Constructor name cannot be empty.")
    config['current_constructors'] = constructors
    
    # Budget
    print("\nBudget:")
    while True:
        try:
            remaining = input("Remaining budget (in millions) [1.0]: ").strip()
            config['remaining_budget'] = float(remaining) if remaining else 1.0
            break
        except ValueError:
            print("  Please enter a valid number.")
    
    # Swaps
    print("\nSwap Limits:")
    while True:
        try:
            step1 = input("Maximum swaps for Step 1 (next race) [2]: ").strip()
            config['step1_swaps'] = int(step1) if step1 else 2
            if config['step1_swaps'] < 0:
                print("  Swaps cannot be negative.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")
    
    while True:
        try:
            step2 = input("Maximum swaps for Step 2 (race after next) [2]: ").strip()
            config['step2_swaps'] = int(step2) if step2 else 2
            if config['step2_swaps'] < 0:
                print("  Swaps cannot be negative.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")
    
    # Weighting scheme
    print("\nVFM Weighting Scheme:")
    print("  1. Equal weights")
    print("  2. Linear decay (older races weighted less)")
    print("  3. Exponential decay")
    print("  4. Moderate decay")
    print("  5. Trend-based (adaptive based on performance trend)")
    
    scheme_map = {
        '1': 'equal',
        '2': 'linear_decay',
        '3': 'exp_decay',
        '4': 'moderate_decay',
        '5': 'trend_based'
    }
    
    while True:
        choice = input("Select scheme (1-5) [5]: ").strip()
        if not choice:
            config['weighting_scheme'] = 'trend_based'
            break
        elif choice in scheme_map:
            config['weighting_scheme'] = scheme_map[choice]
            break
        else:
            print("  Please select a valid option (1-5).")
    
    # Risk tolerance
    print("\nRisk Tolerance:")
    print("  1. Low (prefer consistent performers)")
    print("  2. Medium (balanced approach)")
    print("  3. High (prioritize track-specific performance)")
    
    risk_map = {'1': 'low', '2': 'medium', '3': 'high'}
    
    while True:
        choice = input("Select risk tolerance (1-3) [2]: ").strip()
        if not choice:
            config['risk_tolerance'] = 'medium'
            break
        elif choice in risk_map:
            config['risk_tolerance'] = risk_map[choice]
            break
        else:
            print("  Please select a valid option (1-3).")
    
    # Multiplier
    while True:
        try:
            mult = input("\nMultiplier for best driver [2]: ").strip()
            config['multiplier'] = int(mult) if mult else 2
            if config['multiplier'] < 1:
                print("  Multiplier must be at least 1.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")
    
    # Parallel processing
    parallel = input("\nEnable parallel processing? (y/n) [y]: ").strip().lower()
    config['use_parallel'] = parallel != 'n'
    
    # Confirmation
    print("\n" + "="*80)
    print("Configuration Summary:")
    print("="*80)
    print(f"Base path: {config['base_path']}")
    print(f"Races completed: {config['races_completed']}")
    print(f"Current drivers: {', '.join(config['current_drivers'])}")
    print(f"Current constructors: {', '.join(config['current_constructors'])}")
    print(f"Remaining budget: ${config['remaining_budget']}M")
    print(f"Step 1 swaps: {config['step1_swaps']}")
    print(f"Step 2 swaps: {config['step2_swaps']}")
    print(f"Weighting scheme: {config['weighting_scheme']}")
    print(f"Risk tolerance: {config['risk_tolerance']}")
    print(f"Multiplier: {config['multiplier']}")
    print(f"Parallel processing: {'Enabled' if config['use_parallel'] else 'Disabled'}")
    
    confirm = input("\nProceed with this configuration? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("Configuration cancelled.")
        sys.exit(0)
    
    return config


class F1VFMCalculator:
    """Calculate Value For Money (VFM) scores with outlier removal"""
    
    def __init__(self, config):
        self.config = config
        self.base_path = config['base_path']
        self.scheme = config['weighting_scheme']
        
    def calculate_vfm(self, race_data_file, vfm_data_file, entity_type='driver', weights=None):
        """Calculate VFM scores with outlier removal"""
        # Read and clean data
        race_df = pd.read_csv(race_data_file, skipinitialspace=True)
        race_df.columns = [col.strip() for col in race_df.columns]
        
        if 'Cost' not in race_df.columns:
            raise ValueError("Input data must contain a 'Cost' column.")
        
        # Identify race columns and convert cost
        race_columns = [col for col in race_df.columns if col.startswith('Race')]
        num_races = len(race_columns)
        race_df['Cost_Numeric'] = race_df['Cost'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))))
        
        # Determine entity column
        entity_col = 'Driver' if entity_type.lower() == 'driver' else 'Constructor'
        
        # Remove outliers
        race_df = self._remove_outliers(race_df, entity_col, race_columns)
        
        # Calculate VFM based on scheme
        if self.scheme == 'trend_based':
            race_df = self._calculate_trend_based_vfm(race_df, entity_col, race_columns, num_races)
        else:
            race_df = self._calculate_weighted_vfm(race_df, race_columns, num_races, weights)
        
        # Calculate final VFM
        race_df['VFM'] = (race_df['Weighted_Points'] / race_df['Cost_Numeric']).round(2)
        
        # Create output dataframe
        if entity_type.lower() == 'driver':
            if self.scheme == 'trend_based':
                vfm_df = race_df[['Driver', 'Team', 'Cost', 'VFM', 'Performance_Trend', 'Weights_Used']]
            else:
                vfm_df = race_df[['Driver', 'Team', 'Cost', 'VFM', 'Weights_Used']]
        else:
            if self.scheme == 'trend_based':
                vfm_df = race_df[['Constructor', 'Cost', 'VFM', 'Performance_Trend', 'Weights_Used']]
            else:
                vfm_df = race_df[['Constructor', 'Cost', 'VFM', 'Weights_Used']]
        
        vfm_df = vfm_df.sort_values('VFM', ascending=False)
        vfm_df.to_csv(vfm_data_file, index=False)
        
        return vfm_df
    
    def _remove_outliers(self, df, entity_col, race_columns):
        """Remove race results outside 2 standard deviations"""
        df_clean = df.copy()
        
        for entity in df_clean[entity_col].unique():
            entity_mask = df_clean[entity_col] == entity
            entity_rows = df_clean.loc[entity_mask, race_columns]
            
            # Get all race values for this entity
            if len(entity_rows) > 0:
                entity_data = entity_rows.values
                if entity_data.ndim == 1:
                    entity_data = entity_data.reshape(1, -1)
                
                # Flatten to get all values
                all_values = entity_data.flatten()
                valid_values = all_values[~np.isnan(all_values)]
                
                if len(valid_values) > 0:
                    entity_mean = np.mean(valid_values)
                    entity_std = np.std(valid_values)
                    
                    lower_bound = entity_mean - 2 * entity_std
                    upper_bound = entity_mean + 2 * entity_std
                    
                    # Apply bounds to each race
                    for race_col in race_columns:
                        values = df_clean.loc[entity_mask, race_col].values
                        if len(values) > 0:
                            value = values[0] if np.isscalar(values) or len(values) == 1 else values
                            if np.isscalar(value):
                                if value < lower_bound or value > upper_bound:
                                    df_clean.loc[entity_mask, race_col] = np.nan
                            else:
                                # Handle array case
                                mask = (value < lower_bound) | (value > upper_bound)
                                value[mask] = np.nan
                                df_clean.loc[entity_mask, race_col] = value
        
        return df_clean
    
    def _calculate_trend_based_vfm(self, df, entity_col, race_columns, num_races):
        """Calculate VFM using trend-based weights"""
        entity_weights = {}
        trend_types = {}
        
        for entity in df[entity_col].unique():
            entity_mask = df[entity_col] == entity
            entity_data = df[df[entity_col] == entity]
            points = []
            valid_indices = []
            
            # Safely extract race points
            for i, col in enumerate(race_columns):
                values = entity_data[col].values
                if len(values) > 0:
                    val = values[0] if len(values) == 1 else values.item() if values.size == 1 else values[0]
                    if not np.isnan(val):
                        points.append(val)
                        valid_indices.append(i)
            
            if len(points) >= 3:
                x = np.array(valid_indices)
                coeffs = np.polyfit(x, points, 1)
                slope = coeffs[0]
                trend_threshold = 1.7
                
                full_weights = [np.nan] * num_races
                
                if slope > trend_threshold:
                    base_weights = [0.6 ** (len(valid_indices) - i) for i in range(len(valid_indices))]
                    trend_types[entity] = "Improving"
                elif slope < -trend_threshold:
                    base_weights = [0.7 ** (len(valid_indices) - i) for i in range(len(valid_indices))]
                    trend_types[entity] = "Declining"
                else:
                    base_weights = [0.8 + (0.2 * i / (len(valid_indices) - 1)) for i in range(len(valid_indices))]
                    trend_types[entity] = "Stable"
                
                for i, idx in enumerate(valid_indices):
                    full_weights[idx] = base_weights[i]
                entity_weights[entity] = full_weights
            else:
                full_weights = [np.nan] * num_races
                min_weight = 0.7
                if len(points) == 1:
                    base_weights = [1.0]
                else:
                    base_weights = [min_weight + ((1.0 - min_weight) / (len(points) - 1)) * i for i in range(len(points))]
                
                for i, idx in enumerate(valid_indices):
                    full_weights[idx] = base_weights[i]
                entity_weights[entity] = full_weights
                trend_types[entity] = "Insufficient data"
        
        # Apply weights
        df['Weighted_Points'] = 0.0
        df['Performance_Trend'] = ""
        df['Weights_Used'] = ""
        
        for entity in df[entity_col].unique():
            entity_mask = df[entity_col] == entity
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for i, race_col in enumerate(race_columns):
                race_values = df.loc[entity_mask, race_col].values
                if len(race_values) > 0:
                    race_value = race_values[0] if len(race_values) == 1 else race_values.item() if race_values.size == 1 else race_values[0]
                    weight = entity_weights[entity][i]
                    
                    if not np.isnan(race_value) and not np.isnan(weight):
                        weighted_sum += race_value * weight
                        weight_sum += weight
            
            if weight_sum > 0:
                df.loc[entity_mask, 'Weighted_Points'] = weighted_sum / weight_sum
            
            valid_weights = [w for w in entity_weights[entity] if not np.isnan(w)]
            weight_str = ','.join([f"{w:.2f}" for w in valid_weights])
            df.loc[entity_mask, 'Weights_Used'] = weight_str
            df.loc[entity_mask, 'Performance_Trend'] = trend_types.get(entity, "Unknown")
        
        return df
    
    def _calculate_weighted_vfm(self, df, race_columns, num_races, weights):
        """Calculate VFM using fixed weighting schemes"""
        if weights is None:
            if self.scheme == 'equal':
                weights = [1] * num_races
            elif self.scheme == 'linear_decay':
                weights = [(i + 1) / num_races for i in range(num_races)]
            elif self.scheme == 'exp_decay':
                weights = [max(0, 0.8 ** (num_races - i - 1)) for i in range(num_races)]
            elif self.scheme == 'moderate_decay':
                min_weight = 0.6
                if num_races == 1:
                    weights = [1.0]
                else:
                    weights = [max(0, min_weight + ((1.0 - min_weight) / (num_races - 1)) * i) for i in range(num_races)]
            else:
                weights = [1] * num_races
        
        weights = [max(0, w) for w in weights]
        
        df['Weighted_Points'] = 0.0
        for idx in df.index:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for i, race_col in enumerate(race_columns):
                race_values = df.loc[idx, race_col]
                
                # Handle both scalar and array values
                if hasattr(race_values, '__len__') and not isinstance(race_values, str):
                    race_value = race_values[0] if len(race_values) > 0 else np.nan
                else:
                    race_value = race_values
                
                if not np.isnan(race_value):
                    weighted_sum += race_value * weights[i]
                    weight_sum += weights[i]
            
            if weight_sum > 0:
                df.loc[idx, 'Weighted_Points'] = weighted_sum / weight_sum
        
        df['Weights_Used'] = ""
        return df
    
    def run(self):
        """Run VFM calculations for both drivers and constructors"""
        print("Calculating VFM scores...")
        
        # Calculate driver VFM
        driver_race_file = f'{self.base_path}driver_race_data.csv'
        driver_vfm_file = f'{self.base_path}driver_vfm.csv'
        driver_vfm = self.calculate_vfm(driver_race_file, driver_vfm_file, 'driver')
        
        # Calculate constructor VFM
        constructor_race_file = f'{self.base_path}constructor_race_data.csv'
        constructor_vfm_file = f'{self.base_path}constructor_vfm.csv'
        constructor_vfm = self.calculate_vfm(constructor_race_file, constructor_vfm_file, 'constructor')
        
        print(f"VFM calculation complete. Files saved to {self.base_path}")
        return driver_vfm, constructor_vfm


class F1TrackAffinityCalculator:
    """Calculate track affinity scores based on historical performance with advanced features"""
    
    def __init__(self, config):
        self.config = config
        self.base_path = config['base_path']
        
    def run(self):
        """Run track affinity calculations"""
        print("Calculating track affinities with enhanced model...")
        
        # Load data
        driver_points = pd.read_csv(f'{self.base_path}driver_race_data.csv')
        constructor_points = pd.read_csv(f'{self.base_path}constructor_race_data.csv')
        race_calendar = pd.read_csv(f'{self.base_path}calendar.csv')
        track_characteristics = pd.read_csv(f'{self.base_path}tracks.csv')
        
        # Remove outliers
        race_columns = [col for col in driver_points.columns if col.startswith('Race')]
        constructor_race_columns = [col for col in constructor_points.columns if col.startswith('Race')]
        
        driver_points_clean = self._remove_outliers(driver_points, 'Driver', race_columns)
        constructor_points_clean = self._remove_outliers(constructor_points, 'Constructor', constructor_race_columns)
        
        # Process data
        driver_performance = self._prepare_performance_data(driver_points_clean, race_columns, 'Driver')
        constructor_performance = self._prepare_performance_data(constructor_points_clean, constructor_race_columns, 'Constructor')
        
        # Merge with calendar and track data
        driver_circuit_performance = self._merge_track_data(driver_performance, race_calendar, track_characteristics)
        constructor_circuit_performance = self._merge_track_data(constructor_performance, race_calendar, track_characteristics)
        
        # Encode categorical variables
        driver_perf_encoded, constructor_perf_encoded, track_chars_encoded = self._encode_categoricals(
            driver_circuit_performance, constructor_circuit_performance, track_characteristics
        )
        
        # Add track clustering
        track_clusters = self._cluster_tracks(track_chars_encoded)
        driver_perf_encoded['track_cluster'] = driver_perf_encoded['Circuit'].map(track_clusters)
        constructor_perf_encoded['track_cluster'] = constructor_perf_encoded['Circuit'].map(track_clusters)
        
        # Calculate track difficulty adjustments
        track_difficulty = self._calculate_track_difficulty(driver_perf_encoded, constructor_perf_encoded)
        
        # Apply difficulty adjustments
        driver_perf_encoded = self._apply_difficulty_adjustment(driver_perf_encoded, track_difficulty, 'Driver')
        constructor_perf_encoded = self._apply_difficulty_adjustment(constructor_perf_encoded, track_difficulty, 'Constructor')
        
        # Calculate enhanced affinities
        driver_char_affinity = self._calculate_enhanced_affinity(driver_perf_encoded, 'Driver')
        constructor_char_affinity = self._calculate_enhanced_affinity(constructor_perf_encoded, 'Constructor')
        
        # Convert to DataFrames
        driver_char_affinity_df = pd.DataFrame(driver_char_affinity).T.fillna(0)
        constructor_char_affinity_df = pd.DataFrame(constructor_char_affinity).T.fillna(0)
        
        # Calculate track affinities with enhanced scoring
        driver_track_affinity = self._calculate_enhanced_track_affinity(driver_char_affinity_df, track_chars_encoded, track_clusters)
        constructor_track_affinity = self._calculate_enhanced_track_affinity(constructor_char_affinity_df, track_chars_encoded, track_clusters)
        
        # Create final output
        driver_final_output = self._create_final_output(track_characteristics, driver_track_affinity)
        constructor_final_output = self._create_final_output(track_characteristics, constructor_track_affinity)
        
        # Save outputs
        driver_char_affinity_df.round(3).to_csv(f'{self.base_path}driver_characteristic_affinities.csv')
        driver_final_output.round(3).to_csv(f'{self.base_path}driver_affinity.csv', index=False)
        constructor_char_affinity_df.round(3).to_csv(f'{self.base_path}constructor_characteristic_affinities.csv')
        constructor_final_output.round(3).to_csv(f'{self.base_path}constructor_affinity.csv', index=False)
        
        print(f"Enhanced track affinity calculation complete. Files saved to {self.base_path}")
        return driver_final_output, constructor_final_output
    
    def _remove_outliers(self, df, entity_col, race_columns):
        """Remove outliers from race data"""
        df_clean = df.copy()
        
        for entity in df_clean[entity_col].unique():
            entity_mask = df_clean[entity_col] == entity
            entity_data = df_clean.loc[entity_mask, race_columns].values.flatten()
            
            entity_mean = np.mean(entity_data)
            entity_std = np.std(entity_data)
            
            lower_bound = entity_mean - 2 * entity_std
            upper_bound = entity_mean + 2 * entity_std
            
            for race_col in race_columns:
                value = df_clean.loc[entity_mask, race_col].values[0]
                if value < lower_bound or value > upper_bound:
                    df_clean.loc[entity_mask, race_col] = np.nan
        
        return df_clean
    
    def _prepare_performance_data(self, df, race_columns, entity_type):
        """Prepare performance data in long format"""
        id_vars = ['Driver', 'Team'] if entity_type == 'Driver' else ['Constructor']
        return df.melt(
            id_vars=id_vars,
            value_vars=race_columns,
            var_name='Race',
            value_name='Points'
        )
    
    def _merge_track_data(self, performance_df, race_calendar, track_characteristics):
        """Merge performance data with track information"""
        df = performance_df.merge(race_calendar, on='Race', how='left')
        df = df.merge(track_characteristics, on=['Grand Prix', 'Circuit'], how='left')
        return df
    
    def _encode_categoricals(self, driver_perf, constructor_perf, track_chars):
        """Encode categorical variables"""
        categorical_cols = ['Overtaking Opportunities', 'Track Speed', 'Expected Temperatures']
        label_encoders = {}
        
        track_chars_encoded = track_chars.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            track_chars_encoded[col + '_encoded'] = le.fit_transform(track_chars_encoded[col])
            label_encoders[col] = le
        
        driver_perf_encoded = driver_perf.copy()
        constructor_perf_encoded = constructor_perf.copy()
        
        for col in categorical_cols:
            if col in driver_perf_encoded.columns:
                driver_perf_encoded[col + '_encoded'] = label_encoders[col].transform(driver_perf_encoded[col])
            if col in constructor_perf_encoded.columns:
                constructor_perf_encoded[col + '_encoded'] = label_encoders[col].transform(constructor_perf_encoded[col])
        
        # Add interaction features
        for df in [track_chars_encoded, driver_perf_encoded, constructor_perf_encoded]:
            if all(col in df.columns for col in ['Corners', 'Length (km)', 'Track Speed_encoded']):
                df['Corners_per_km'] = df['Corners'] / df['Length (km)']
                df['High_speed_corners'] = df['Track Speed_encoded'] * df['Corners']
                df['Technical_score'] = df['Corners_per_km'] * (2 - df['Track Speed_encoded'])
        
        return driver_perf_encoded, constructor_perf_encoded, track_chars_encoded
    
    def _cluster_tracks(self, track_chars_encoded):
        """Cluster tracks based on characteristics"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Features for clustering
        features = ['Corners', 'Length (km)', 'Overtaking Opportunities_encoded', 
                   'Track Speed_encoded', 'Expected Temperatures_encoded',
                   'Corners_per_km', 'High_speed_corners', 'Technical_score']
        
        X = track_chars_encoded[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        n_clusters = min(5, len(track_chars_encoded) // 3)  # Ensure reasonable cluster size
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create circuit to cluster mapping
        circuit_clusters = dict(zip(track_chars_encoded['Circuit'], clusters))
        
        return circuit_clusters
    
    def _calculate_track_difficulty(self, driver_perf, constructor_perf):
        """Calculate track difficulty based on performance variance"""
        track_difficulty = {}
        
        # Combine all performance data
        all_performances = []
        
        for circuit in driver_perf['Circuit'].unique():
            circuit_data = driver_perf[driver_perf['Circuit'] == circuit]
            valid_points = circuit_data.dropna(subset=['Points'])['Points'].values
            
            if len(valid_points) > 3:
                # Calculate difficulty metrics
                mean_points = np.mean(valid_points)
                std_points = np.std(valid_points)
                cv = std_points / mean_points if mean_points > 0 else 0
                
                # Higher CV = more difficult/unpredictable track
                track_difficulty[circuit] = {
                    'mean': mean_points,
                    'std': std_points,
                    'cv': cv,
                    'difficulty_score': cv
                }
            else:
                track_difficulty[circuit] = {
                    'mean': np.mean(valid_points) if len(valid_points) > 0 else 0,
                    'std': 0,
                    'cv': 0,
                    'difficulty_score': 0
                }
        
        return track_difficulty
    
    def _apply_difficulty_adjustment(self, perf_df, track_difficulty, entity_type):
        """Apply track difficulty adjustments to performance data"""
        perf_df = perf_df.copy()
        perf_df['adjusted_points'] = perf_df['Points']
        perf_df['difficulty_score'] = 0
        
        for circuit, difficulty in track_difficulty.items():
            circuit_mask = perf_df['Circuit'] == circuit
            if difficulty['std'] > 0:
                # Z-score normalization
                perf_df.loc[circuit_mask, 'adjusted_points'] = (
                    (perf_df.loc[circuit_mask, 'Points'] - difficulty['mean']) / difficulty['std']
                )
            perf_df.loc[circuit_mask, 'difficulty_score'] = difficulty['difficulty_score']
        
        return perf_df
    
    def _calculate_enhanced_affinity(self, df, entity_col):
        """Calculate enhanced affinity with multiple improvements"""
        df_valid = df.dropna(subset=['Points', 'adjusted_points'])
        
        # All features to consider
        base_chars = ['Corners', 'Length (km)', 'Overtaking Opportunities_encoded',
                     'Track Speed_encoded', 'Expected Temperatures_encoded']
        interaction_chars = ['Corners_per_km', 'High_speed_corners', 'Technical_score']
        all_chars = base_chars + interaction_chars
        
        entity_affinities = {}
        
        for entity in df_valid[entity_col].unique():
            entity_data = df_valid[df_valid[entity_col] == entity]
            
            if len(entity_data) < 3:
                continue
            
            affinity_scores = {}
            
            # Time weights (exponential decay)
            n_races = len(entity_data)
            time_weights = np.exp(-0.1 * (n_races - np.arange(n_races)))
            time_weights = time_weights / time_weights.sum()
            
            # Base linear affinities
            for char in all_chars:
                if char in entity_data.columns and entity_data[char].notna().sum() > 2:
                    valid_mask = entity_data[char].notna() & entity_data['adjusted_points'].notna()
                    valid_data = entity_data.loc[valid_mask]
                    
                    if len(valid_data) >= 3:
                        # Weighted correlation
                        char_values = valid_data[char].values
                        points_values = valid_data['adjusted_points'].values
                        
                        # Ensure we have the right number of weights
                        weights = time_weights[-len(valid_data):] if len(time_weights) >= len(valid_data) else time_weights
                        
                        try:
                            # Calculate weighted correlation safely
                            if len(char_values) > 1 and len(points_values) > 1:
                                weighted_cov = np.cov(char_values, points_values, aweights=weights)[0, 1]
                                weighted_var_char = np.var(char_values, aweights=weights)
                                weighted_var_points = np.var(points_values, aweights=weights)
                                
                                if weighted_var_char > 0 and weighted_var_points > 0:
                                    correlation = weighted_cov / np.sqrt(weighted_var_char * weighted_var_points)
                                else:
                                    correlation = 0
                            else:
                                correlation = 0
                        except Exception:
                            correlation = 0
                        
                        # Confidence weight based on sample size
                        confidence = min(len(valid_data) / 7, 1.0)
                        affinity_scores[char] = correlation * confidence
                    else:
                        affinity_scores[char] = 0
                else:
                    affinity_scores[char] = 0
            
            # Non-linear affinities (polynomial features)
            for char in base_chars[:2]:  # Only for continuous variables (Corners, Length)
                if char in entity_data.columns:
                    valid_mask = entity_data[char].notna() & entity_data['adjusted_points'].notna()
                    valid_data = entity_data.loc[valid_mask]
                    
                    if len(valid_data) >= 5:  # Need more data for polynomial
                        char_values = valid_data[char].values
                        points_values = valid_data['adjusted_points'].values
                        
                        # Quadratic term
                        char_squared = char_values ** 2
                        try:
                            correlation_squared = np.corrcoef(char_squared, points_values)[0, 1]
                            if not np.isnan(correlation_squared):
                                affinity_scores[f'{char}_squared'] = correlation_squared * 0.5  # Lower weight
                        except Exception:
                            pass
            
            # Cluster-based affinity
            if 'track_cluster' in entity_data.columns:
                cluster_performance = entity_data.groupby('track_cluster')['adjusted_points'].agg(['mean', 'count'])
                for cluster in cluster_performance.index:
                    if cluster_performance.loc[cluster, 'count'] >= 2:
                        affinity_scores[f'cluster_{cluster}'] = float(cluster_performance.loc[cluster, 'mean'])
            
            # Consistency factor
            consistency = 1 / (1 + entity_data['adjusted_points'].std()) if len(entity_data) > 1 else 1
            affinity_scores['consistency'] = float(consistency)
            
            entity_affinities[entity] = affinity_scores
        
        return entity_affinities
    
    def _calculate_enhanced_track_affinity(self, char_affinity_df, track_chars_encoded, track_clusters):
        """Calculate track affinity scores with enhanced model"""
        base_chars = ['Corners', 'Length (km)', 'Overtaking Opportunities_encoded',
                     'Track Speed_encoded', 'Expected Temperatures_encoded']
        interaction_chars = ['Corners_per_km', 'High_speed_corners', 'Technical_score']
        
        track_affinity_scores = {}
        
        for entity in char_affinity_df.index:
            entity_scores = {}
            entity_affinities = char_affinity_df.loc[entity]
            
            for _, track in track_chars_encoded.iterrows():
                # Base linear score
                base_score = 0
                base_weight = 0
                
                for char in base_chars + interaction_chars:
                    if char in track and char in entity_affinities:
                        char_value = track[char]
                        entity_affinity = entity_affinities[char]
                        
                        # Normalize characteristic values
                        if char == 'Corners':
                            normalized_value = (char_value - 10) / 17
                        elif char == 'Length (km)':
                            normalized_value = (char_value - 3.337) / 3.667
                        elif char == 'Corners_per_km':
                            normalized_value = min(char_value / 10, 1)
                        else:
                            max_val = track_chars_encoded[char].max()
                            normalized_value = char_value / max_val if max_val > 0 else 0
                        
                        contribution = entity_affinity * normalized_value
                        base_score += contribution
                        base_weight += abs(entity_affinity)
                
                # Non-linear score
                nonlinear_score = 0
                nonlinear_weight = 0
                
                for char in ['Corners', 'Length (km)']:
                    squared_key = f'{char}_squared'
                    if squared_key in entity_affinities and char in track:
                        char_value = track[char]
                        # Normalize and square
                        if char == 'Corners':
                            normalized_value = (char_value - 10) / 17
                        else:
                            normalized_value = (char_value - 3.337) / 3.667
                        
                        squared_value = normalized_value ** 2
                        contribution = entity_affinities[squared_key] * squared_value
                        nonlinear_score += contribution
                        nonlinear_weight += abs(entity_affinities[squared_key])
                
                # Cluster score
                cluster_score = 0
                if track['Circuit'] in track_clusters:
                    cluster = track_clusters[track['Circuit']]
                    cluster_key = f'cluster_{cluster}'
                    if cluster_key in entity_affinities:
                        cluster_score = entity_affinities[cluster_key]
                
                # Consistency bonus
                consistency_bonus = entity_affinities.get('consistency', 0)
                
                # Combine scores with weights
                total_weight = base_weight + nonlinear_weight
                if total_weight > 0:
                    linear_component = (base_score / base_weight) if base_weight > 0 else 0
                    nonlinear_component = (nonlinear_score / nonlinear_weight) if nonlinear_weight > 0 else 0
                    
                    # Weighted combination
                    final_score = (
                        0.5 * linear_component +
                        0.2 * nonlinear_component +
                        0.2 * cluster_score +
                        0.1 * consistency_bonus
                    )
                else:
                    final_score = 0
                
                entity_scores[track['Circuit']] = final_score
            
            track_affinity_scores[entity] = entity_scores
        
        return track_affinity_scores
    
    def _create_final_output(self, track_characteristics, track_affinity):
        """Create final output table with affinities"""
        track_affinity_df = pd.DataFrame(track_affinity).T.fillna(0)
        final_output = track_characteristics[['Grand Prix', 'Circuit']].copy()
        
        for entity in track_affinity_df.index:
            final_output[f'{entity}_affinity'] = final_output['Circuit'].map(
                track_affinity_df.loc[entity].to_dict()
            )
        
        return final_output.fillna(0)


class F1TeamOptimizer:
    """Optimize F1 Fantasy team selections for upcoming races"""
    
    def __init__(self, config):
        self.config = config
        self.base_path = config['base_path']
        self.multiplier = config['multiplier']
        self.risk_tolerance = config['risk_tolerance']
        
        # Data containers
        self.drivers_df = None
        self.constructors_df = None
        self.track_affinity_df = None
        self.constructor_affinity_df = None
        self.race_calendar_df = None
        self.max_budget = None
        self.current_team_cost = None
        
        # Weight adjustments
        self._set_risk_weights()
        
        # Caches
        self.step1_cache = {}
        self.step2_cache = {}
        
        # Performance stats
        self.performance_stats = {
            'patterns_evaluated': 0,
            'cache_hits': 0,
            'optimization_time': 0
        }
    
    def _set_risk_weights(self):
        """Set weights based on risk tolerance"""
        if self.risk_tolerance == 'low':
            self.affinity_weight = 0.5
            self.vfm_weight = 1.5
        elif self.risk_tolerance == 'high':
            self.affinity_weight = 1.5
            self.vfm_weight = 0.5
        else:
            self.affinity_weight = 1.0
            self.vfm_weight = 1.0
    
    def load_data(self):
        """Load all required data files"""
        try:
            self.drivers_df = pd.read_csv(f'{self.base_path}driver_vfm.csv')
            self.constructors_df = pd.read_csv(f'{self.base_path}constructor_vfm.csv')
            self.track_affinity_df = pd.read_csv(f'{self.base_path}driver_affinity.csv')
            self.constructor_affinity_df = pd.read_csv(f'{self.base_path}constructor_affinity.csv')
            self.race_calendar_df = pd.read_csv(f'{self.base_path}calendar.csv')
            
            # Parse costs
            self.drivers_df['Cost_Value'] = self.drivers_df['Cost'].str.replace('[$M]', '', regex=True).astype(float)
            self.constructors_df['Cost_Value'] = self.constructors_df['Cost'].str.replace('[$M]', '', regex=True).astype(float)
            
            # Determine target races
            self.step1_race = f"Race{self.config['races_completed'] + 1}"
            self.step2_race = f"Race{self.config['races_completed'] + 2}"
            
            # Get race information
            step1_info = self.race_calendar_df[self.race_calendar_df['Race'] == self.step1_race]
            step2_info = self.race_calendar_df[self.race_calendar_df['Race'] == self.step2_race]
            
            if step1_info.empty or step2_info.empty:
                raise ValueError("Race information not found")
            
            self.step1_circuit = step1_info.iloc[0]['Circuit']
            self.step1_gp = step1_info.iloc[0]['Grand Prix']
            self.step2_circuit = step2_info.iloc[0]['Circuit']
            self.step2_gp = step2_info.iloc[0]['Grand Prix']
            
            # Prepare affinities
            self._prepare_track_affinities()
            
            # Calculate budget
            cd_cost = self.drivers_df[self.drivers_df['Driver'].isin(self.config['current_drivers'])]['Cost_Value'].sum()
            cc_cost = self.constructors_df[self.constructors_df['Constructor'].isin(self.config['current_constructors'])]['Cost_Value'].sum()
            self.current_team_cost = cd_cost + cc_cost
            self.max_budget = self.current_team_cost + self.config['remaining_budget']
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _prepare_track_affinities(self):
        """Prepare track-specific VFM adjustments"""
        self.drivers_df['VFM_Original'] = self.drivers_df['VFM'].copy()
        self.constructors_df['VFM_Original'] = self.constructors_df['VFM'].copy()
        
        # Get affinity columns
        driver_affinity_cols = [col for col in self.track_affinity_df.columns if col.endswith('_affinity')]
        constructor_affinity_cols = [col for col in self.constructor_affinity_df.columns if col.endswith('_affinity')]
        
        # Apply affinities for both steps
        for step, circuit in [(1, self.step1_circuit), (2, self.step2_circuit)]:
            # Driver affinities
            circuit_row = self.track_affinity_df[self.track_affinity_df['Circuit'] == circuit]
            if not circuit_row.empty:
                for col in driver_affinity_cols:
                    driver_name = col.replace('_affinity', '')
                    if driver_name in self.drivers_df['Driver'].values:
                        affinity = circuit_row.iloc[0][col] if pd.notna(circuit_row.iloc[0][col]) else 0
                        driver_idx = self.drivers_df[self.drivers_df['Driver'] == driver_name].index[0]
                        
                        # Apply weighted adjustment
                        base_vfm = self.drivers_df.loc[driver_idx, 'VFM_Original']
                        adjusted_vfm = base_vfm * (1 + affinity * self.affinity_weight)
                        
                        self.drivers_df.loc[driver_idx, f'Step {step}_Affinity'] = affinity
                        self.drivers_df.loc[driver_idx, f'Step {step}_VFM'] = adjusted_vfm
            
            # Constructor affinities
            circuit_row = self.constructor_affinity_df[self.constructor_affinity_df['Circuit'] == circuit]
            if not circuit_row.empty:
                for col in constructor_affinity_cols:
                    constructor_name = col.replace('_affinity', '')
                    if constructor_name in self.constructors_df['Constructor'].values:
                        affinity = circuit_row.iloc[0][col] if pd.notna(circuit_row.iloc[0][col]) else 0
                        constructor_idx = self.constructors_df[self.constructors_df['Constructor'] == constructor_name].index[0]
                        
                        # Apply weighted adjustment
                        base_vfm = self.constructors_df.loc[constructor_idx, 'VFM_Original']
                        adjusted_vfm = base_vfm * (1 + affinity * self.affinity_weight)
                        
                        self.constructors_df.loc[constructor_idx, f'Step {step}_Affinity'] = affinity
                        self.constructors_df.loc[constructor_idx, f'Step {step}_VFM'] = adjusted_vfm
    
    def _get_team_data(self, drivers, constructors, step):
        """Get team data with step-specific VFM"""
        driver_df = self.drivers_df[self.drivers_df['Driver'].isin(drivers)].copy()
        constructor_df = self.constructors_df[self.constructors_df['Constructor'].isin(constructors)].copy()
        
        driver_df['VFM'] = driver_df[f'Step {step}_VFM']
        constructor_df['VFM'] = constructor_df[f'Step {step}_VFM']
        
        return driver_df, constructor_df
    
    def evaluate_team(self, drivers, constructors, step):
        """Evaluate team performance"""
        cache = self.step1_cache if step == 1 else self.step2_cache
        key = "|".join(sorted(drivers)) + "#" + "|".join(sorted(constructors))
        
        if key in cache:
            self.performance_stats['cache_hits'] += 1
            return cache[key]
        
        driver_df, constructor_df = self._get_team_data(drivers, constructors, step)
        
        cost = driver_df['Cost_Value'].sum() + constructor_df['Cost_Value'].sum()
        if cost > self.max_budget:
            cache[key] = (-1, -1, cost, None)
            return -1, -1, cost, None
        
        base_vfm = driver_df['VFM'].sum() + constructor_df['VFM'].sum()
        count = len(drivers) + len(constructors)
        
        best_ppm = -1
        best_total_points = -1
        best_driver = None
        
        for _, row in driver_df.iterrows():
            boosted_vfm = base_vfm + (self.multiplier - 1) * row['VFM']
            ppm = boosted_vfm / count
            total_points = ppm * cost
            
            if total_points > best_total_points:
                best_total_points = total_points
                best_ppm = ppm
                best_driver = row['Driver']
        
        cache[key] = (best_total_points, best_ppm, cost, best_driver)
        return best_total_points, best_ppm, cost, best_driver
    
    def generate_swap_patterns(self, current_drivers, current_constructors, max_swaps):
        """Generate all valid swap patterns"""
        patterns = []
        
        # Pre-calculate minimum costs for pruning
        min_driver_cost = self.drivers_df['Cost_Value'].min()
        min_constructor_cost = self.constructors_df['Cost_Value'].min()
        
        for total_swaps in range(max_swaps + 1):
            for driver_swaps in range(min(total_swaps + 1, len(current_drivers) + 1)):
                constructor_swaps = total_swaps - driver_swaps
                if constructor_swaps > len(current_constructors):
                    continue
                
                # Quick budget check
                potential_savings = driver_swaps * min_driver_cost + constructor_swaps * min_constructor_cost
                if self.current_team_cost - potential_savings > self.max_budget:
                    continue
                
                for drivers_out in itertools.combinations(current_drivers, driver_swaps):
                    for constructors_out in itertools.combinations(current_constructors, constructor_swaps):
                        patterns.append((drivers_out, constructors_out))
        
        return patterns
    
    def evaluate_swap_pattern(self, pattern, current_drivers, current_constructors, 
                            available_drivers, available_constructors, step):
        """Evaluate a specific swap pattern"""
        drivers_out, constructors_out = pattern
        
        best_result = {
            'points': -1,
            'ppm': -1,
            'swaps': [],
            'drivers': current_drivers,
            'constructors': current_constructors,
            'cost': 0,
            'boost_driver': None
        }
        
        # Try all combinations
        for new_drivers in itertools.combinations(available_drivers, len(drivers_out)):
            candidate_drivers = [d for d in current_drivers if d not in drivers_out] + list(new_drivers)
            
            for new_constructors in itertools.combinations(available_constructors, len(constructors_out)):
                candidate_constructors = [c for c in current_constructors if c not in constructors_out] + list(new_constructors)
                
                points, ppm, cost, boost_driver = self.evaluate_team(candidate_drivers, candidate_constructors, step)
                
                if points > best_result['points']:
                    swaps = [('Driver', old, new) for old, new in zip(drivers_out, new_drivers)]
                    swaps += [('Constructor', old, new) for old, new in zip(constructors_out, new_constructors)]
                    
                    best_result = {
                        'points': points,
                        'ppm': ppm,
                        'swaps': swaps,
                        'drivers': candidate_drivers,
                        'constructors': candidate_constructors,
                        'cost': cost,
                        'boost_driver': boost_driver
                    }
        
        self.performance_stats['patterns_evaluated'] += 1
        return best_result
    
    def optimize_step(self, current_drivers, current_constructors, max_swaps, step):
        """Optimize team for a specific step"""
        patterns = self.generate_swap_patterns(current_drivers, current_constructors, max_swaps)
        available_drivers = [d for d in self.drivers_df['Driver'] if d not in current_drivers]
        available_constructors = [c for c in self.constructors_df['Constructor'] if c not in current_constructors]
        
        best_result = {
            'points': -1,
            'ppm': -1,
            'swaps': [],
            'drivers': current_drivers,
            'constructors': current_constructors,
            'cost': 0,
            'boost_driver': None
        }
        
        # Evaluate baseline
        base_points, base_ppm, base_cost, base_boost = self.evaluate_team(current_drivers, current_constructors, step)
        if base_points > 0:
            best_result = {
                'points': base_points,
                'ppm': base_ppm,
                'swaps': [],
                'drivers': current_drivers,
                'constructors': current_constructors,
                'cost': base_cost,
                'boost_driver': base_boost
            }
        
        # Evaluate all patterns
        if self.config['use_parallel'] and len(patterns) > 50:
            # Parallel processing for large pattern sets
            with Pool(processes=cpu_count()) as pool:
                results = pool.starmap(
                    self.evaluate_swap_pattern,
                    [(p, current_drivers, current_constructors, available_drivers, available_constructors, step) 
                     for p in patterns]
                )
                
                for result in results:
                    if result['points'] > best_result['points']:
                        best_result = result
        else:
            # Sequential processing
            for pattern in patterns:
                result = self.evaluate_swap_pattern(
                    pattern, current_drivers, current_constructors, 
                    available_drivers, available_constructors, step
                )
                if result['points'] > best_result['points']:
                    best_result = result
        
        return best_result
    
    def run_dual_step_optimization(self):
        """Run the complete two-step optimization"""
        start_time = time.time()
        
        current_drivers = self.config['current_drivers']
        current_constructors = self.config['current_constructors']
        
        # Get baseline performance
        base_s1 = self.evaluate_team(current_drivers, current_constructors, 1)
        base_s2 = self.evaluate_team(current_drivers, current_constructors, 2)
        
        print(f"\nOptimizing for {self.step1_race} ({self.step1_circuit}) and {self.step2_race} ({self.step2_circuit})")
        print(f"Current team - Step 1: {base_s1[0]:.1f} points, Step 2: {base_s2[0]:.1f} points")
        
        # Track best overall result
        best_overall = {
            'step1_result': None,
            'step2_result': None,
            'final_points': base_s2[0],
            'step1_points': base_s1[0]
        }
        
        # Try all Step 1 possibilities
        print("\nEvaluating Step 1 options...")
        step1_patterns = self.generate_swap_patterns(current_drivers, current_constructors, self.config['step1_swaps'])
        
        for i, pattern in enumerate(step1_patterns):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(step1_patterns)} patterns", end='\r')
            
            # Evaluate Step 1
            step1_result = self.evaluate_swap_pattern(
                pattern, current_drivers, current_constructors,
                [d for d in self.drivers_df['Driver'] if d not in current_drivers],
                [c for c in self.constructors_df['Constructor'] if c not in current_constructors],
                1
            )
            
            if step1_result['points'] <= 0:
                continue
            
            # Evaluate Step 2 from this Step 1 team
            step2_result = self.optimize_step(
                step1_result['drivers'], 
                step1_result['constructors'],
                self.config['step2_swaps'],
                2
            )
            
            # Check if this is the best overall
            if (step2_result['points'] > best_overall['final_points'] or
                (step2_result['points'] == best_overall['final_points'] and 
                 step1_result['points'] > best_overall['step1_points'])):
                best_overall = {
                    'step1_result': step1_result,
                    'step2_result': step2_result,
                    'final_points': step2_result['points'],
                    'step1_points': step1_result['points']
                }
        
        self.performance_stats['optimization_time'] = time.time() - start_time
        
        return best_overall, base_s1, base_s2
    
    def print_results(self, optimization_result, base_s1, base_s2):
        """Print optimization results"""
        step1_result = optimization_result['step1_result']
        step2_result = optimization_result['step2_result']
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        
        # Step 1 results
        if step1_result and step1_result['swaps']:
            print(f"\nStep 1 - {self.step1_race} at {self.step1_circuit}:")
            print("Swaps:")
            for entity_type, old, new in step1_result['swaps']:
                if entity_type == 'Driver':
                    old_data = self.drivers_df[self.drivers_df['Driver'] == old].iloc[0]
                    new_data = self.drivers_df[self.drivers_df['Driver'] == new].iloc[0]
                    print(f"  {old} → {new} (${old_data['Cost_Value']}M → ${new_data['Cost_Value']}M)")
                else:
                    old_data = self.constructors_df[self.constructors_df['Constructor'] == old].iloc[0]
                    new_data = self.constructors_df[self.constructors_df['Constructor'] == new].iloc[0]
                    print(f"  {old} → {new} (${old_data['Cost_Value']}M → ${new_data['Cost_Value']}M)")
            
            print(f"Expected points: {step1_result['points']:.1f} (+{step1_result['points'] - base_s1[0]:.1f})")
            print(f"Boost driver: {step1_result['boost_driver']}")
        else:
            print(f"\nStep 1 - No changes recommended")
        
        # Step 2 results
        if step2_result:
            print(f"\nStep 2 - {self.step2_race} at {self.step2_circuit}:")
            if step2_result['swaps']:
                print("Additional swaps:")
                for entity_type, old, new in step2_result['swaps']:
                    if entity_type == 'Driver':
                        old_data = self.drivers_df[self.drivers_df['Driver'] == old].iloc[0]
                        new_data = self.drivers_df[self.drivers_df['Driver'] == new].iloc[0]
                        print(f"  {old} → {new} (${old_data['Cost_Value']}M → ${new_data['Cost_Value']}M)")
                    else:
                        old_data = self.constructors_df[self.constructors_df['Constructor'] == old].iloc[0]
                        new_data = self.constructors_df[self.constructors_df['Constructor'] == new].iloc[0]
                        print(f"  {old} → {new} (${old_data['Cost_Value']}M → ${new_data['Cost_Value']}M)")
            else:
                print("No additional swaps")
            
            print(f"Expected points: {step2_result['points']:.1f} (+{step2_result['points'] - base_s2[0]:.1f})")
            print(f"Boost driver: {step2_result['boost_driver']}")
            print(f"Final budget used: ${step2_result['cost']:.1f}M / ${self.max_budget:.1f}M")
        
        # Summary
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total improvement: {optimization_result['final_points'] - base_s2[0]:.1f} points")
        print(f"Patterns evaluated: {self.performance_stats['patterns_evaluated']:,}")
        print(f"Cache hits: {self.performance_stats['cache_hits']:,}")
        print(f"Time taken: {self.performance_stats['optimization_time']:.1f}s")
    
    def save_results(self, optimization_result):
        """Save optimization results to JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'races': {
                'step1': {'race': self.step1_race, 'circuit': self.step1_circuit},
                'step2': {'race': self.step2_race, 'circuit': self.step2_circuit}
            },
            'optimization': optimization_result,
            'performance_stats': self.performance_stats
        }
        
        filename = f"{self.base_path}optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def run(self):
        """Run the complete optimization"""
        if not self.load_data():
            return False
        
        optimization_result, base_s1, base_s2 = self.run_dual_step_optimization()
        self.print_results(optimization_result, base_s1, base_s2)
        self.save_results(optimization_result)
        
        return True


def main():
    """Main execution function"""
    print("F1 Fantasy Optimizer Suite")
    print("="*80)
    
    # Check if there's a saved configuration
    CONFIG = None
    config_file = None
    
    # Try common paths
    possible_paths = [
        '/content/drive/MyDrive/F1 Fantasy/last_config.json',
        './last_config.json',
        'last_config.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            config_file = path
            break
    
    if config_file:
        use_saved = input(f"\nFound saved configuration at {config_file}. Use it? (y/n) [n]: ").strip().lower()
        if use_saved == 'y':
            try:
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                
                # Recalculate races_completed
                races_completed = get_races_completed(saved_config['base_path'])
                if races_completed is not None:
                    saved_config['races_completed'] = races_completed
                    print(f"\nLoaded configuration with {races_completed} races completed.")
                    CONFIG = saved_config
                else:
                    print("Error reading race data. Starting fresh configuration.")
            except Exception as e:
                print(f"Error loading saved configuration: {e}")
                print("Starting fresh configuration.")
    
    # Get configuration from user if not loaded
    if CONFIG is None:
        CONFIG = get_user_configuration()
    
    # Step 1: Calculate VFM
    print("\n\nStep 1: Calculating VFM scores...")
    vfm_calculator = F1VFMCalculator(CONFIG)
    driver_vfm, constructor_vfm = vfm_calculator.run()
    
    # Step 2: Calculate Track Affinities
    print("\nStep 2: Calculating track affinities...")
    affinity_calculator = F1TrackAffinityCalculator(CONFIG)
    driver_affinities, constructor_affinities = affinity_calculator.run()
    
    # Step 3: Optimize Team
    print("\nStep 3: Optimizing team selection...")
    optimizer = F1TeamOptimizer(CONFIG)
    optimizer.run()
    
    print("\n" + "="*80)
    print("Optimization complete!")
    
    # Save configuration for future reference
    config_file = f"{CONFIG['base_path']}last_config.json"
    with open(config_file, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\nConfiguration saved to {config_file}")
    
    # Ask if user wants to load this config next time
    print("\nTip: To reuse this configuration next time, select 'y' when prompted at startup.")


if __name__ == "__main__":
    main()
