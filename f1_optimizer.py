#!/usr/bin/env python3
"""
F1 Fantasy Optimizer Suite
Complete pipeline for VFM calculation, track affinity analysis, and team optimization
Enhanced with improved affinity calculations and FP2 pace integration
"""

import pandas as pd
import numpy as np
import re
import sys
import os
import json
import itertools
import time
import requests
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
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


def get_expected_race_pace(session_key):
    """
    Calculate the expected race pace for each driver after a given practice session.
    Parameters:
    session_key (int): The unique identifier for the practice session.
    Returns:
    pandas.DataFrame: A DataFrame containing driver numbers and their average lap times.
    """
    # Fetch lap data for the session
    url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    lap_data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(lap_data)
    if df.empty:
        raise Exception("No lap data found for the given session_key.")
    
    # Filter out in-laps, out-laps, and laps with missing lap_time
    df = df[(df['lap_type'] == 'FLYING') & (df['lap_time'].notnull())]
    if df.empty:
        raise Exception("No valid flying laps found for the given session_key.")
    
    # Calculate average lap time per driver
    avg_lap_times = df.groupby('driver_number')['lap_time'].mean().reset_index()
    avg_lap_times.rename(columns={'lap_time': 'average_lap_time'}, inplace=True)
    
    # Sort by average lap time
    avg_lap_times.sort_values(by='average_lap_time', inplace=True)
    
    return avg_lap_times


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
    
    # FP2 Pace Integration
    print("\nFP2 Race Pace Integration:")
    use_fp2 = input("Use FP2 race pace data? (y/n) [n]: ").strip().lower()
    config['use_fp2_pace'] = use_fp2 == 'y'
    
    if config['use_fp2_pace']:
        while True:
            try:
                session_key = input("FP2 session key from OpenF1 API: ").strip()
                config['fp2_session_key'] = int(session_key)
                break
            except ValueError:
                print("  Please enter a valid session key number.")
        
        print("\nPace Weight (how much to weight current form vs historical):")
        print("  1. Conservative (20% current, 80% historical)")
        print("  2. Balanced (30% current, 70% historical)")  
        print("  3. Aggressive (40% current, 60% historical)")
        
        weight_map = {'1': 0.20, '2': 0.30, '3': 0.40}
        choice = input("Select pace weight (1-3) [2]: ").strip()
        config['pace_weight'] = weight_map.get(choice, 0.30)
        
        print("\nPace Modifier Type:")
        print("  1. Conservative (0.8x to 1.2x range)")
        print("  2. Aggressive (0.6x to 1.4x range)")
        
        modifier_map = {'1': 'conservative', '2': 'aggressive'}
        choice = input("Select modifier type (1-2) [1]: ").strip()
        config['pace_modifier_type'] = modifier_map.get(choice, 'conservative')
    
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
    if config['use_fp2_pace']:
        print(f"FP2 session key: {config['fp2_session_key']}")
        print(f"Pace weight: {config['pace_weight']}")
        print(f"Modifier type: {config['pace_modifier_type']}")
    print(f"Parallel processing: {'Enabled' if config['use_parallel'] else 'Disabled'}")
    
    confirm = input("\nProceed with this configuration? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("Configuration cancelled.")
        sys.exit(0)
    
    return config


class F1VFMCalculator:
    """Calculate Value For Money (VFM) scores with outlier removal and FP2 pace integration"""
    
    def __init__(self, config):
        self.config = config
        self.base_path = config['base_path']
        self.scheme = config['weighting_scheme']
        self.use_fp2_pace = config.get('use_fp2_pace', False)
        self.fp2_session_key = config.get('fp2_session_key', None)
        self.pace_weight = config.get('pace_weight', 0.25)
        self.pace_modifier_type = config.get('pace_modifier_type', 'conservative')
        
    def calculate_vfm(self, race_data_file, vfm_data_file, entity_type='driver', weights=None):
        """Calculate VFM scores with outlier removal and optional FP2 pace integration"""
        # Calculate base VFM
        race_df = self._calculate_base_vfm(race_data_file, entity_type, weights)
        
        # Apply FP2 pace modifier if enabled and we have session key
        if self.use_fp2_pace and self.fp2_session_key:
            try:
                pace_data = get_expected_race_pace(self.fp2_session_key)
                race_df = self._apply_pace_modifiers(race_df, pace_data, entity_type)
                print(f"Applied FP2 pace modifiers from session {self.fp2_session_key}")
            except Exception as e:
                print(f"Warning: Could not apply FP2 pace data: {e}")
                print("Continuing with historical VFM only...")
        
        # Save results
        race_df.to_csv(vfm_data_file, index=False)
        return race_df
    
    def _calculate_base_vfm(self, race_data_file, entity_type, weights):
        """Calculate base VFM scores with outlier removal"""
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
                result_df = race_df[['Driver', 'Team', 'Cost', 'VFM', 'Performance_Trend', 'Weights_Used']]
            else:
                result_df = race_df[['Driver', 'Team', 'Cost', 'VFM', 'Weights_Used']]
        else:
            if self.scheme == 'trend_based':
                result_df = race_df[['Constructor', 'Cost', 'VFM', 'Performance_Trend', 'Weights_Used']]
            else:
                result_df = race_df[['Constructor', 'Cost', 'VFM', 'Weights_Used']]
        
        result_df = result_df.sort_values('VFM', ascending=False)
        return result_df
    
    def _apply_pace_modifiers(self, race_df, pace_data, entity_type):
        """Apply FP2 pace-based VFM modifications"""
        # Calculate pace scores
        pace_scores = self._calculate_pace_scores(pace_data)
        
        # Load driver/constructor mapping
        driver_mapping = self._load_driver_number_mapping()
        if driver_mapping is None:
            print("Warning: No driver mapping available. Skipping FP2 pace integration.")
            return race_df
        
        pace_scores = pace_scores.merge(driver_mapping, on='driver_number', how='left')
        
        entity_col = 'Driver' if entity_type.lower() == 'driver' else 'Constructor'
        
        # Add pace tracking columns
        race_df['VFM_Pre_Pace'] = race_df['VFM'].copy()
        race_df['Pace_Score'] = 0.0
        race_df['Pace_Modifier'] = 1.0
        
        # Apply modifiers
        for _, row in pace_scores.iterrows():
            if entity_type.lower() == 'driver':
                entity_name = row.get('driver_name', '')
            else:
                entity_name = row.get('team_name', '')
            
            if pd.notna(entity_name) and entity_name in race_df[entity_col].values:
                mask = race_df[entity_col] == entity_name
                
                pace_score = row['pace_score']
                modifier = self._calculate_pace_modifier(pace_score)
                
                race_df.loc[mask, 'Pace_Score'] = pace_score
                race_df.loc[mask, 'Pace_Modifier'] = modifier
                race_df.loc[mask, 'VFM'] = race_df.loc[mask, 'VFM'] * modifier
        
        return race_df
    
    def _calculate_pace_scores(self, avg_lap_times_df):
        """Convert lap times to relative performance scores"""
        df = avg_lap_times_df.copy()
        
        # Get fastest lap time as baseline
        fastest_time = df['average_lap_time'].min()
        
        # Calculate percentage gap to fastest
        df['gap_to_fastest'] = (df['average_lap_time'] - fastest_time) / fastest_time * 100
        
        # Convert to performance score (0-100 scale, 100 = fastest)
        max_gap = df['gap_to_fastest'].max()
        if max_gap > 0:
            df['pace_score'] = 100 * (1 - df['gap_to_fastest'] / max_gap)
        else:
            df['pace_score'] = 100.0
        
        return df
    
    def _calculate_pace_modifier(self, pace_score):
        """Convert pace score to VFM modifier"""
        if self.pace_modifier_type == 'aggressive':
            # Aggressive: 0.6x to 1.4x range
            base_modifier = 0.6 + (pace_score / 100) * 0.8
        else:
            # Conservative: 0.8x to 1.2x range (recommended)
            base_modifier = 0.8 + (pace_score / 100) * 0.4
        
        # Apply risk-based adjustment
        risk_tolerance = self.config.get('risk_tolerance', 'medium')
        
        if risk_tolerance == 'low':
            # Dampen the pace effect (more conservative)
            dampening_factor = 0.5
            modifier = 1.0 + (base_modifier - 1.0) * dampening_factor
        elif risk_tolerance == 'high':
            # Amplify the pace effect (more aggressive)
            amplification_factor = 1.5
            modifier = 1.0 + (base_modifier - 1.0) * amplification_factor
        else:
            # Medium risk - use base modifier
            modifier = base_modifier
        
        return modifier
    
    def _load_driver_number_mapping(self):
        """Load driver number to name mapping"""
        mapping_file = os.path.join(self.base_path, 'driver_mapping.csv')
        
        if os.path.exists(mapping_file):
            try:
                mapping_df = pd.read_csv(mapping_file)
                return mapping_df
            except Exception as e:
                print(f"Error loading driver mapping: {e}")
        
        # If no mapping file, try to create basic mapping from driver data
        try:
            driver_data = pd.read_csv(os.path.join(self.base_path, 'driver_race_data.csv'))
            # This would need to be enhanced with actual driver numbers
            # For now, return None to skip pace integration
            print("No driver mapping file found. Create 'driver_mapping.csv' with columns: driver_number, driver_name, team_name")
            return None
        except:
            return None
    
    def _remove_outliers(self, df, entity_col, race_columns):
        """Remove race results outside 2 standard deviations"""
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
    
    def _calculate_trend_based_vfm(self, df, entity_col, race_columns, num_races):
        """Calculate VFM using trend-based weights"""
        entity_weights = {}
        trend_types = {}
        
        for entity in df[entity_col].unique():
            entity_data = df[df[entity_col] == entity]
            points = []
            valid_indices = []
            
            for i, col in enumerate(race_columns):
                val = entity_data[col].values[0]
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
                race_value = df.loc[entity_mask, race_col].values[0]
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
                race_value = df.loc[idx, race_col]
                
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
    """Calculate track affinity scores based on historical performance with enhanced algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.base_path = config['base_path']
        
    def run(self):
        """Run track affinity calculations"""
        print("Calculating track affinities with enhanced algorithms...")
        
        # Load data
        driver_points = pd.read_csv(f'{self.base_path}driver_race_data.csv')
        constructor_points = pd.read_csv(f'{self.base_path}constructor_race_data.csv')
        race_calendar = pd.read_csv(f'{self.base_path}calendar.csv')
        track_characteristics = pd.read_csv(f'{self.base_path}tracks.csv')
        
        # Remove outliers
        race_columns = [col for col in driver_points.columns if col.startswith('Race')]
        constructor_race_columns = [col for col in constructor_points.columns if col.startswith('Race')]
        
        driver_points_clean = self._remove_outliers_advanced(driver_points, 'Driver', race_columns)
        constructor_points_clean = self._remove_outliers_advanced(constructor_points, 'Constructor', constructor_race_columns)
        
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
        
        # Calculate characteristic importance weights
        char_importance = self._calculate_characteristic_importance(track_chars_encoded)
        
        # Calculate enhanced affinities
        driver_char_affinity = self._calculate_enhanced_affinity(driver_perf_encoded, 'Driver', char_importance)
        constructor_char_affinity = self._calculate_enhanced_affinity(constructor_perf_encoded, 'Constructor', char_importance)
        
        # Convert to DataFrames
        driver_char_affinity_df = pd.DataFrame(driver_char_affinity).T.fillna(0)
        constructor_char_affinity_df = pd.DataFrame(constructor_char_affinity).T.fillna(0)
        
        # Calculate track affinities
        driver_track_affinity = self._calculate_track_affinity(driver_char_affinity_df, track_chars_encoded)
        constructor_track_affinity = self._calculate_track_affinity(constructor_char_affinity_df, track_chars_encoded)
        
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
    
    def _remove_outliers_advanced(self, df, entity_col, race_columns):
        """Enhanced outlier detection using IQR and rolling statistics"""
        df_clean = df.copy()
        
        for entity in df_clean[entity_col].unique():
            entity_mask = df_clean[entity_col] == entity
            entity_data = df_clean.loc[entity_mask, race_columns].values.flatten()
            entity_data = entity_data[~np.isnan(entity_data)]
            
            if len(entity_data) < 4:
                continue
                
            # Use IQR method for more robust outlier detection
            Q1 = np.percentile(entity_data, 25)
            Q3 = np.percentile(entity_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Also consider rolling variance for seasonal outliers
            if len(entity_data) >= 6:
                rolling_std = pd.Series(entity_data).rolling(window=3, center=True).std()
                dynamic_threshold = rolling_std.mean() * 2.5
                
                # Apply more conservative bounds if rolling variance suggests high volatility
                if not np.isnan(dynamic_threshold):
                    entity_mean = np.mean(entity_data)
                    lower_bound = max(lower_bound, entity_mean - dynamic_threshold)
                    upper_bound = min(upper_bound, entity_mean + dynamic_threshold)
            
            for race_col in race_columns:
                value = df_clean.loc[entity_mask, race_col].values[0]
                if not np.isnan(value) and (value < lower_bound or value > upper_bound):
                    df_clean.loc[entity_mask, race_col] = np.nan
        
        return df_clean
    
    def _prepare_performance_data(self, df, race_columns, entity_type):
        """Prepare performance data in long format with race ordering"""
        id_vars = ['Driver', 'Team'] if entity_type == 'Driver' else ['Constructor']
        melted_df = df.melt(
            id_vars=id_vars,
            value_vars=race_columns,
            var_name='Race',
            value_name='Points'
        )
        
        # Extract race number for proper ordering
        melted_df['Race_Number'] = melted_df['Race'].str.extract('(\d+)').astype(int)
        melted_df = melted_df.sort_values(['Race_Number'])
        
        return melted_df
    
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
        
        return driver_perf_encoded, constructor_perf_encoded, track_chars_encoded
    
    def _calculate_characteristic_importance(self, track_chars_encoded):
        """Calculate dynamic weights for each characteristic based on variance and impact"""
        char_cols = ['Corners', 'Length (km)', 'Overtaking Opportunities_encoded',
                     'Track Speed_encoded', 'Expected Temperatures_encoded']
        
        importance_weights = {}
        variances = []
        
        for char in char_cols:
            variance = np.var(track_chars_encoded[char])
            variances.append(variance)
        
        max_variance = max(variances)
        
        for i, char in enumerate(char_cols):
            # Higher variance = more discriminative = higher weight
            # Normalize to 0.5-2.0 range
            if max_variance > 0:
                importance_weights[char] = 0.5 + (variances[i] / max_variance) * 1.5
            else:
                importance_weights[char] = 1.0
        
        return importance_weights
    
    def _weighted_correlation(self, x, y, weights):
        """Calculate weighted correlation coefficient"""
        if len(x) != len(y) or len(x) != len(weights):
            return np.nan
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(weights))
        if valid_mask.sum() < 2:
            return np.nan
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        w_valid = weights[valid_mask]
        
        # Calculate weighted means
        w_sum = np.sum(w_valid)
        if w_sum == 0:
            return np.nan
            
        x_mean = np.sum(w_valid * x_valid) / w_sum
        y_mean = np.sum(w_valid * y_valid) / w_sum
        
        # Calculate weighted covariance and variances
        cov = np.sum(w_valid * (x_valid - x_mean) * (y_valid - y_mean)) / w_sum
        var_x = np.sum(w_valid * (x_valid - x_mean)**2) / w_sum
        var_y = np.sum(w_valid * (y_valid - y_mean)**2) / w_sum
        
        # Calculate correlation
        if var_x == 0 or var_y == 0:
            return np.nan
        
        return cov / np.sqrt(var_x * var_y)
    
    def _calculate_enhanced_affinity(self, df, entity_col, char_importance):
        """Calculate enhanced entity affinity to track characteristics"""
        df_valid = df.dropna(subset=['Points'])
        
        char_cols = ['Corners', 'Length (km)', 'Overtaking Opportunities_encoded',
                     'Track Speed_encoded', 'Expected Temperatures_encoded']
        
        # Add interaction terms
        interaction_pairs = [
            ('Corners', 'Track Speed_encoded'),
            ('Length (km)', 'Overtaking Opportunities_encoded'),
            ('Track Speed_encoded', 'Expected Temperatures_encoded')
        ]
        
        entity_char_affinity = {}
        
        for entity in df_valid[entity_col].unique():
            entity_data = df_valid[df_valid[entity_col] == entity].copy()
            
            if len(entity_data) < 3:
                continue
            
            # Sort by race number for time-weighted analysis
            entity_data = entity_data.sort_values('Race_Number')
            
            affinity_scores = {}
            
            # Create time weights (recent races weighted more heavily)
            n_races = len(entity_data)
            time_weights = np.exp(np.linspace(-1.5, 0, n_races))
            
            # Analyze base characteristics
            for char in char_cols:
                if char in entity_data.columns and entity_data[char].notna().sum() >= 2:
                    char_data = entity_data[char].values
                    points_data = entity_data['Points'].values
                    
                    # Multi-scale temporal analysis
                    long_term_corr = self._calculate_robust_correlation(char_data, points_data)
                    
                    # Short-term analysis (last 40% of races)
                    short_term_threshold = max(2, int(0.4 * n_races))
                    if n_races >= short_term_threshold:
                        recent_char = char_data[-short_term_threshold:]
                        recent_points = points_data[-short_term_threshold:]
                        short_term_corr = self._calculate_robust_correlation(recent_char, recent_points)
                        
                        # Blend correlations (70% long-term, 30% short-term)
                        if not np.isnan(short_term_corr):
                            blended_corr = 0.7 * long_term_corr + 0.3 * short_term_corr
                        else:
                            blended_corr = long_term_corr
                    else:
                        blended_corr = long_term_corr
                    
                    # Apply confidence weighting using bootstrap
                    confidence_weight = self._calculate_confidence_weight(char_data, points_data)
                    
                    # Apply characteristic importance weighting
                    importance_weight = char_importance.get(char, 1.0)
                    
                    final_affinity = blended_corr * confidence_weight * importance_weight
                    affinity_scores[char] = final_affinity
                else:
                    affinity_scores[char] = 0
            
            # Calculate interaction effects
            for char1, char2 in interaction_pairs:
                if (char1 in entity_data.columns and char2 in entity_data.columns and 
                    entity_data[char1].notna().sum() >= 2 and entity_data[char2].notna().sum() >= 2):
                    
                    # Create interaction term
                    interaction_data = entity_data[char1].values * entity_data[char2].values
                    points_data = entity_data['Points'].values
                    
                    # Calculate interaction correlation
                    interaction_corr = self._calculate_robust_correlation(interaction_data, points_data)
                    confidence_weight = self._calculate_confidence_weight(interaction_data, points_data)
                    
                    # Store interaction effect (weighted less than main effects)
                    interaction_name = f"{char1}_x_{char2}"
                    affinity_scores[interaction_name] = interaction_corr * confidence_weight * 0.5
            
            entity_char_affinity[entity] = affinity_scores
        
        return entity_char_affinity
    
    def _calculate_robust_correlation(self, x, y):
        """Calculate robust correlation using multiple methods"""
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if valid_mask.sum() < 2:
            return 0
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Test multiple relationship types
        correlations = []
        
        # Linear correlation
        linear_corr = np.corrcoef(x_valid, y_valid)[0, 1]
        if not np.isnan(linear_corr):
            correlations.append(linear_corr)
        
        # Quadratic relationship
        if len(x_valid) >= 3:
            x_squared = x_valid ** 2
            quad_corr = np.corrcoef(x_squared, y_valid)[0, 1]
            if not np.isnan(quad_corr):
                correlations.append(quad_corr)
        
        # Threshold relationship (performance above/below median)
        if len(np.unique(x_valid)) > 2:
            median_x = np.median(x_valid)
            threshold_mask = x_valid > median_x
            if len(np.unique(threshold_mask)) > 1:
                threshold_corr = np.corrcoef(threshold_mask.astype(int), y_valid)[0, 1]
                if not np.isnan(threshold_corr):
                    correlations.append(threshold_corr)
        
        # Return strongest relationship (by absolute value)
        if correlations:
            return max(correlations, key=abs)
        else:
            return 0
    
    def _calculate_confidence_weight(self, x, y):
        """Calculate confidence weight using bootstrap resampling"""
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if valid_mask.sum() < 3:
            return 0.1  # Low confidence for insufficient data
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Bootstrap resampling for confidence intervals
        bootstrap_correlations = []
        n_bootstrap = min(50, len(x_valid) * 10)  # Adaptive bootstrap sample size
        
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(len(x_valid), size=len(x_valid), replace=True)
            sample_x = x_valid[sample_indices]
            sample_y = y_valid[sample_indices]
            
            sample_corr = np.corrcoef(sample_x, sample_y)[0, 1]
            if not np.isnan(sample_corr):
                bootstrap_correlations.append(sample_corr)
        
        if len(bootstrap_correlations) < 5:
            return 0.1
        
        # Calculate confidence interval width
        confidence_interval = np.percentile(bootstrap_correlations, [25, 75])
        confidence_width = confidence_interval[1] - confidence_interval[0]
        
        # Convert width to confidence weight (narrow CI = higher confidence)
        # Normalize to 0.1-1.0 range
        max_expected_width = 1.0  # Maximum reasonable correlation width
        confidence_weight = max(0.1, 1.0 - min(confidence_width / max_expected_width, 0.9))
        
        return confidence_weight
    
    def _calculate_track_affinity(self, char_affinity_df, track_chars_encoded):
        """Calculate entity affinity for each track with interaction effects"""
        base_char_cols = ['Corners', 'Length (km)', 'Overtaking Opportunities_encoded',
                         'Track Speed_encoded', 'Expected Temperatures_encoded']
        
        # Include interaction terms in calculation
        interaction_cols = [col for col in char_affinity_df.columns if '_x_' in col]
        all_char_cols = base_char_cols + interaction_cols
        
        track_affinity_scores = {}
        
        for entity in char_affinity_df.index:
            entity_scores = {}
            entity_affinities = char_affinity_df.loc[entity]
            
            for _, track in track_chars_encoded.iterrows():
                track_score = 0
                total_weight = 0
                
                # Calculate base characteristic contributions
                for char in base_char_cols:
                    if char in track and char in entity_affinities:
                        char_value = track[char]
                        entity_affinity = entity_affinities[char]
                        
                        # Normalize characteristic values
                        if char == 'Corners':
                            normalized_value = (char_value - 10) / (27 - 10)
                        elif char == 'Length (km)':
                            normalized_value = (char_value - 3.337) / (7.004 - 3.337)
                        else:
                            max_encoded = track_chars_encoded[char].max()
                            normalized_value = char_value / max_encoded if max_encoded > 0 else 0
                        
                        # Clip to [0, 1] range
                        normalized_value = np.clip(normalized_value, 0, 1)
                        
                        contribution = entity_affinity * normalized_value
                        track_score += contribution
                        total_weight += abs(entity_affinity)
                
                # Add interaction effects
                for interaction_col in interaction_cols:
                    if interaction_col in entity_affinities:
                        # Parse interaction column name
                        char1, char2 = interaction_col.split('_x_')
                        
                        if char1 in track and char2 in track:
                            # Calculate interaction value
                            char1_value = track[char1]
                            char2_value = track[char2]
                            
                            # Normalize both characteristics
                            if char1 == 'Corners':
                                norm1 = (char1_value - 10) / (27 - 10)
                            elif char1 == 'Length (km)':
                                norm1 = (char1_value - 3.337) / (7.004 - 3.337)
                            else:
                                max1 = track_chars_encoded[char1].max()
                                norm1 = char1_value / max1 if max1 > 0 else 0
                            
                            if char2 == 'Corners':
                                norm2 = (char2_value - 10) / (27 - 10)
                            elif char2 == 'Length (km)':
                                norm2 = (char2_value - 3.337) / (7.004 - 3.337)
                            else:
                                max2 = track_chars_encoded[char2].max()
                                norm2 = char2_value / max2 if max2 > 0 else 0
                            
                            norm1 = np.clip(norm1, 0, 1)
                            norm2 = np.clip(norm2, 0, 1)
                            
                            interaction_value = norm1 * norm2
                            entity_interaction_affinity = entity_affinities[interaction_col]
                            
                            contribution = entity_interaction_affinity * interaction_value
                            track_score += contribution
                            total_weight += abs(entity_interaction_affinity)
                
                # Calculate final normalized score
                if total_weight > 0:
                    final_score = track_score / total_weight
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
        
        # FP2 pace information if used
        if self.config.get('use_fp2_pace', False):
            print("\n" + "-"*80)
            print("FP2 PACE INTEGRATION")
            print("-"*80)
            print(f"Session key: {self.config.get('fp2_session_key', 'N/A')}")
            print(f"Pace weight: {self.config.get('pace_weight', 0.25)}")
            print(f"Modifier type: {self.config.get('pace_modifier_type', 'conservative')}")
            
            # Show pace adjustments for current drivers
            print("\nPace adjustments for current team:")
            for driver in self.config['current_drivers']:
                driver_row = self.drivers_df[self.drivers_df['Driver'] == driver]
                if not driver_row.empty and 'Pace_Score' in driver_row.columns:
                    pace_score = driver_row.iloc[0].get('Pace_Score', 0)
                    pace_modifier = driver_row.iloc[0].get('Pace_Modifier', 1.0)
                    if pace_score > 0:
                        print(f"  {driver}: Pace score {pace_score:.1f}, VFM modifier {pace_modifier:.2f}x")
    
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
    print("F1 Fantasy Optimizer Suite - Enhanced Edition with FP2 Integration")
    print("=" * 80)
    
    try:
        # Get user configuration
        config = get_user_configuration()
        
        # Initialize components
        vfm_calculator = F1VFMCalculator(config)
        affinity_calculator = F1TrackAffinityCalculator(config)
        team_optimizer = F1TeamOptimizer(config)
        
        # Run pipeline
        print("\nStarting optimization pipeline...")
        
        # Step 1: Calculate VFM scores
        print("\n" + "="*50)
        print("STEP 1: VFM CALCULATION")
        print("="*50)
        driver_vfm, constructor_vfm = vfm_calculator.run()
        
        # Step 2: Calculate track affinities
        print("\n" + "="*50)
        print("STEP 2: TRACK AFFINITY ANALYSIS")
        print("="*50)
        driver_affinity, constructor_affinity = affinity_calculator.run()
        
        # Step 3: Optimize team
        print("\n" + "="*50)
        print("STEP 3: TEAM OPTIMIZATION")
        print("="*50)
        success = team_optimizer.run()
        
        if success:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"All results saved to: {config['base_path']}")
            print("\nGenerated files:")
            print("- driver_vfm.csv (VFM scores for drivers)")
            print("- constructor_vfm.csv (VFM scores for constructors)")
            print("- driver_characteristic_affinities.csv (Driver characteristic correlations)")
            print("- driver_affinity.csv (Driver track affinities)")
            print("- constructor_characteristic_affinities.csv (Constructor characteristic correlations)")
            print("- constructor_affinity.csv (Constructor track affinities)")
            print("- optimization_[timestamp].json (Optimization results)")
            
            if config.get('use_fp2_pace', False):
                print("\nFP2 Integration:")
                print("- Real-time pace data integrated into VFM calculations")
                print("- Create 'driver_mapping.csv' for enhanced pace integration")
        else:
            print("\nOptimization failed. Please check your data files and configuration.")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
