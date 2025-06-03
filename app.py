import os
import json
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import re
from werkzeug.utils import secure_filename
import shutil
from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer, get_races_completed, get_expected_race_pace

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['DEFAULT_DATA_FOLDER'] = 'default_data'
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEFAULT_DATA_FOLDER'], exist_ok=True)

# Store session data (in production, use proper session management)
sessions = {}


def get_data_folder(session_id=None):
    """Get the appropriate data folder - either session-specific or default"""
    if session_id and session_id in sessions:
        return sessions[session_id]['folder']
    elif has_default_data():
        return app.config['DEFAULT_DATA_FOLDER'] + '/'
    return None


def has_default_data():
    """Check if default data files exist"""
    required_files = [
        'driver_race_data.csv',
        'constructor_race_data.csv',
        'calendar.csv',
        'tracks.csv'
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(app.config['DEFAULT_DATA_FOLDER'], file)):
            return False
    return True


def has_driver_mapping():
    """Check if driver mapping file exists"""
    mapping_path = os.path.join(app.config['DEFAULT_DATA_FOLDER'], 'driver_mapping.csv')
    return os.path.exists(mapping_path)


def load_default_data():
    """Load default data if available"""
    if not has_default_data():
        return None

    try:
        # Calculate races completed
        races_completed = get_races_completed(app.config['DEFAULT_DATA_FOLDER'] + '/')

        # Read drivers and constructors
        driver_df = pd.read_csv(os.path.join(app.config['DEFAULT_DATA_FOLDER'], 'driver_race_data.csv'))
        constructor_df = pd.read_csv(os.path.join(app.config['DEFAULT_DATA_FOLDER'], 'constructor_race_data.csv'))

        drivers_list = sorted(driver_df['Driver'].unique().tolist())
        constructors_list = sorted(constructor_df['Constructor'].unique().tolist())

        return {
            'races_completed': races_completed,
            'drivers': drivers_list,
            'constructors': constructors_list,
            'has_driver_mapping': has_driver_mapping()
        }
    except Exception as e:
        print(f"Error loading default data: {e}")
        return None


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/check_default_data')
def check_default_data():
    """Check if default data exists"""
    default_data = load_default_data()
    return jsonify({
        'has_default': default_data is not None,
        'data': default_data
    })


@app.route('/check_driver_mapping')
def check_driver_mapping():
    """Check if driver mapping file exists"""
    return jsonify({
        'exists': has_driver_mapping()
    })


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        # Check if we should update default data
        update_default = request.form.get('update_default', 'false') == 'true'

        session_id = request.form.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))

        # Determine target folder
        if update_default:
            target_folder = app.config['DEFAULT_DATA_FOLDER']
            print("Updating default data files...")
        else:
            target_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            os.makedirs(target_folder, exist_ok=True)

        required_files = [
            'driver_race_data.csv',
            'constructor_race_data.csv',
            'calendar.csv',
            'tracks.csv'
        ]

        uploaded_files = []
        for file_key in request.files:
            file = request.files[file_key]
            if file and file.filename:
                # Use the required filename, not the uploaded filename
                for required_file in required_files:
                    if required_file in file_key:
                        filename = required_file
                        break
                else:
                    filename = secure_filename(file.filename)

                filepath = os.path.join(target_folder, filename)
                file.save(filepath)
                uploaded_files.append(filename)

        # Check if all required files are present
        missing_files = [f for f in required_files if f not in uploaded_files]
        if missing_files:
            return jsonify({
                'success': False,
                'message': f'Missing required files: {", ".join(missing_files)}'
            })

        # Calculate races completed
        races_completed = get_races_completed(target_folder + '/')

        # Read drivers and constructors for selection
        driver_df = pd.read_csv(os.path.join(target_folder, 'driver_race_data.csv'))
        constructor_df = pd.read_csv(os.path.join(target_folder, 'constructor_race_data.csv'))

        drivers_list = sorted(driver_df['Driver'].unique().tolist())
        constructors_list = sorted(constructor_df['Constructor'].unique().tolist())

        # Store session info
        if not update_default:
            sessions[session_id] = {
                'folder': target_folder,
                'races_completed': races_completed,
                'drivers': drivers_list,
                'constructors': constructors_list
            }

        return jsonify({
            'success': True,
            'session_id': session_id if not update_default else 'default',
            'races_completed': races_completed,
            'drivers': drivers_list,
            'constructors': constructors_list,
            'updated_default': update_default,
            'has_driver_mapping': has_driver_mapping() if update_default else False
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error uploading files: {str(e)}'
        })


@app.route('/upload_driver_mapping', methods=['POST'])
def upload_driver_mapping():
    """Handle driver mapping file upload"""
    try:
        if 'driver_mapping' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No driver mapping file provided'
            })

        file = request.files['driver_mapping']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            })

        # Save to default data folder
        mapping_path = os.path.join(app.config['DEFAULT_DATA_FOLDER'], 'driver_mapping.csv')
        file.save(mapping_path)

        # Validate the file
        try:
            mapping_df = pd.read_csv(mapping_path)
            required_columns = ['driver_number', 'driver_name', 'team_name']
            if not all(col in mapping_df.columns for col in required_columns):
                os.remove(mapping_path)
                return jsonify({
                    'success': False,
                    'message': f'Driver mapping file must contain columns: {", ".join(required_columns)}'
                })

            # Additional validation
            if mapping_df.empty:
                os.remove(mapping_path)
                return jsonify({
                    'success': False,
                    'message': 'Driver mapping file is empty'
                })

            # Check for required data types
            if not pd.api.types.is_numeric_dtype(mapping_df['driver_number']):
                os.remove(mapping_path)
                return jsonify({
                    'success': False,
                    'message': 'driver_number column must contain numeric values'
                })

        except Exception as e:
            if os.path.exists(mapping_path):
                os.remove(mapping_path)
            return jsonify({
                'success': False,
                'message': f'Invalid CSV file: {str(e)}'
            })

        return jsonify({
            'success': True,
            'message': f'Driver mapping file uploaded successfully. Found {len(mapping_df)} driver mappings.',
            'driver_count': len(mapping_df)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error uploading driver mapping: {str(e)}'
        })


@app.route('/test_fp2_connection', methods=['POST'])
def test_fp2_connection():
    """Test FP2 API connection with given session key"""
    try:
        data = request.json
        session_key = data.get('session_key')

        if not session_key:
            return jsonify({
                'success': False,
                'message': 'Session key is required'
            })

        try:
            session_key = int(session_key)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'message': 'Session key must be a number'
            })

        # Test the API connection
        pace_data = get_expected_race_pace(session_key)

        if pace_data.empty:
            return jsonify({
                'success': False,
                'message': f'No pace data found for session {session_key}'
            })

        return jsonify({
            'success': True,
            'message': f'Successfully connected to FP2 session {session_key}',
            'driver_count': len(pace_data),
            'fastest_driver': pace_data.iloc[0]['driver_number'] if not pace_data.empty else None,
            'session_key': session_key
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error connecting to FP2 API: {str(e)}'
        })


@app.route('/optimize', methods=['POST'])
def optimize():
    """Run the optimization with optional FP2 pace integration"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')

        # Get data folder
        if session_id == 'default':
            data_folder = app.config['DEFAULT_DATA_FOLDER'] + '/'
            if not has_default_data():
                return jsonify({
                    'success': False,
                    'message': 'No default data available. Please upload files first.'
                })
            # Load default data info
            default_info = load_default_data()
            races_completed = default_info['races_completed']
        else:
            if session_id not in sessions:
                return jsonify({
                    'success': False,
                    'message': 'Session not found. Please upload files first.'
                })
            data_folder = sessions[session_id]['folder'] + '/'
            races_completed = sessions[session_id]['races_completed']

        # Validate FP2 configuration if enabled
        use_fp2_pace = data.get('use_fp2_pace', False)
        fp2_session_key = data.get('fp2_session_key')
        pace_weight = data.get('pace_weight', 0.25)
        pace_modifier_type = data.get('pace_modifier_type', 'conservative')

        if use_fp2_pace:
            if not fp2_session_key:
                return jsonify({
                    'success': False,
                    'message': 'FP2 session key is required when using pace data.'
                })
            try:
                fp2_session_key = int(fp2_session_key)
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'message': 'Invalid FP2 session key. Must be a number.'
                })

            # Check if driver mapping exists
            if not has_driver_mapping():
                return jsonify({
                    'success': False,
                    'message': 'Driver mapping file is required for FP2 integration. Please upload it first.'
                })

        # Build configuration
        config = {
            'base_path': data_folder,
            'races_completed': races_completed,
            'current_drivers': data['current_drivers'],
            'current_constructors': data['current_constructors'],
            'remaining_budget': float(data['remaining_budget']),
            'step1_swaps': int(data['step1_swaps']),
            'step2_swaps': int(data['step2_swaps']),
            'weighting_scheme': data['weighting_scheme'],
            'risk_tolerance': data['risk_tolerance'],
            'multiplier': int(data['multiplier']),
            'use_parallel': False,  # Disable for web app
            # FP2 configuration
            'use_fp2_pace': use_fp2_pace,
            'fp2_session_key': fp2_session_key,
            'pace_weight': float(pace_weight),
            'pace_modifier_type': pace_modifier_type
        }

        results = {
            'status': 'running',
            'progress': []
        }

        # Step 1: Calculate VFM (with optional FP2 integration)
        if config['use_fp2_pace']:
            results['progress'].append(f'Fetching FP2 pace data from session {config["fp2_session_key"]}...')
        results['progress'].append('Calculating VFM scores...')

        vfm_calculator = F1VFMCalculator(config)
        driver_vfm, constructor_vfm = vfm_calculator.run()
        results['progress'].append('VFM calculation complete')

        # Step 2: Calculate Track Affinities
        results['progress'].append('Calculating track affinities...')
        affinity_calculator = F1TrackAffinityCalculator(config)
        driver_affinities, constructor_affinities = affinity_calculator.run()
        results['progress'].append('Track affinity calculation complete')

        # Step 3: Optimize Team
        results['progress'].append('Optimizing team selection...')
        optimizer = F1TeamOptimizer(config)

        # Load data
        if not optimizer.load_data():
            return jsonify({
                'success': False,
                'message': 'Error loading optimization data'
            })

        # Run optimization
        optimization_result, base_s1, base_s2 = optimizer.run_dual_step_optimization()

        # Format results
        step1_result = optimization_result['step1_result']
        step2_result = optimization_result['step2_result']

        results['optimization'] = {
            'step1': {
                'race': optimizer.step1_race,
                'circuit': optimizer.step1_circuit,
                'swaps': step1_result['swaps'] if step1_result else [],
                'expected_points': step1_result['points'] if step1_result else base_s1[0],
                'improvement': (step1_result['points'] - base_s1[0]) if step1_result else 0,
                'boost_driver': step1_result['boost_driver'] if step1_result else base_s1[3],
                'team': {
                    'drivers': step1_result['drivers'] if step1_result else config['current_drivers'],
                    'constructors': step1_result['constructors'] if step1_result else config['current_constructors']
                }
            },
            'step2': {
                'race': optimizer.step2_race,
                'circuit': optimizer.step2_circuit,
                'swaps': step2_result['swaps'] if step2_result else [],
                'expected_points': step2_result['points'] if step2_result else base_s2[0],
                'improvement': (step2_result['points'] - base_s2[0]) if step2_result else 0,
                'boost_driver': step2_result['boost_driver'] if step2_result else base_s2[3],
                'team': {
                    'drivers': step2_result['drivers'] if step2_result else config['current_drivers'],
                    'constructors': step2_result['constructors'] if step2_result else config['current_constructors']
                },
                'budget_used': step2_result['cost'] if step2_result else base_s2[2],
                'budget_remaining': optimizer.max_budget - (step2_result['cost'] if step2_result else base_s2[2])
            },
            'summary': {
                'total_improvement': optimization_result['final_points'] - base_s2[0],
                'patterns_evaluated': optimizer.performance_stats['patterns_evaluated'],
                'optimization_time': optimizer.performance_stats['optimization_time']
            }
        }

        # Add FP2 information if used
        if config['use_fp2_pace']:
            results['optimization']['fp2_info'] = {
                'session_key': config['fp2_session_key'],
                'pace_weight': config['pace_weight'],
                'modifier_type': config['pace_modifier_type'],
                'applied': True
            }

            # Add pace adjustment information for current drivers
            pace_adjustments = []
            for driver in config['current_drivers']:
                driver_row = optimizer.drivers_df[optimizer.drivers_df['Driver'] == driver]
                if not driver_row.empty:
                    pace_score = driver_row.iloc[0].get('Pace_Score', 0)
                    pace_modifier = driver_row.iloc[0].get('Pace_Modifier', 1.0)
                    vfm_original = driver_row.iloc[0].get('VFM_Pre_Pace', driver_row.iloc[0].get('VFM', 0))
                    vfm_adjusted = driver_row.iloc[0].get('VFM', 0)

                    if pace_score > 0:
                        pace_adjustments.append({
                            'driver': driver,
                            'pace_score': round(pace_score, 1),
                            'pace_modifier': round(pace_modifier, 3),
                            'vfm_original': round(vfm_original, 2),
                            'vfm_adjusted': round(vfm_adjusted, 2)
                        })

            results['optimization']['fp2_info']['pace_adjustments'] = pace_adjustments

        results['status'] = 'complete'
        results['success'] = True

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error during optimization: {str(e)}'
        })


@app.route('/statistics')
def statistics():
    """Render the statistics page"""
    return render_template('statistics.html')


@app.route('/api/statistics')
def get_statistics():
    """Get comprehensive statistics for all drivers and constructors"""
    try:
        # Check if we have default data
        data_folder = get_data_folder('default')
        if not data_folder:
            return jsonify({
                'success': False,
                'message': 'No data available. Please upload data files first.'
            })

        # Load all necessary data
        driver_race_df = pd.read_csv(os.path.join(data_folder, 'driver_race_data.csv'))
        constructor_race_df = pd.read_csv(os.path.join(data_folder, 'constructor_race_data.csv'))
        calendar_df = pd.read_csv(os.path.join(data_folder, 'calendar.csv'))
        tracks_df = pd.read_csv(os.path.join(data_folder, 'tracks.csv'))

        # Load affinity data if available
        driver_affinity_path = os.path.join(data_folder, 'driver_affinity.csv')
        constructor_affinity_path = os.path.join(data_folder, 'constructor_affinity.csv')
        driver_char_affinity_path = os.path.join(data_folder, 'driver_characteristic_affinities.csv')
        constructor_char_affinity_path = os.path.join(data_folder, 'constructor_characteristic_affinities.csv')

        # Run calculations if affinity files don't exist
        if not all(os.path.exists(p) for p in [driver_affinity_path, constructor_affinity_path,
                                               driver_char_affinity_path, constructor_char_affinity_path]):
            config = {
                'base_path': data_folder,
                'races_completed': get_races_completed(data_folder),
                'weighting_scheme': 'trend_based',
                'use_fp2_pace': False  # Disable FP2 for statistics calculation
            }

            # Calculate VFM
            vfm_calc = F1VFMCalculator(config)
            vfm_calc.run()

            # Calculate affinities
            affinity_calc = F1TrackAffinityCalculator(config)
            affinity_calc.run()

        # Load the calculated data
        driver_vfm_df = pd.read_csv(os.path.join(data_folder, 'driver_vfm.csv'))
        constructor_vfm_df = pd.read_csv(os.path.join(data_folder, 'constructor_vfm.csv'))
        driver_affinity_df = pd.read_csv(driver_affinity_path)
        constructor_affinity_df = pd.read_csv(constructor_affinity_path)
        driver_char_affinity_df = pd.read_csv(driver_char_affinity_path, index_col=0)
        constructor_char_affinity_df = pd.read_csv(constructor_char_affinity_path, index_col=0)

        # Process driver statistics
        driver_stats = process_driver_statistics(
            driver_race_df, driver_vfm_df, driver_affinity_df,
            driver_char_affinity_df, calendar_df, tracks_df
        )

        # Process constructor statistics
        constructor_stats = process_constructor_statistics(
            constructor_race_df, constructor_vfm_df, constructor_affinity_df,
            constructor_char_affinity_df, calendar_df, tracks_df
        )

        return jsonify({
            'success': True,
            'drivers': driver_stats,
            'constructors': constructor_stats,
            'races_completed': get_races_completed(data_folder)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error calculating statistics: {str(e)}'
        })


def process_driver_statistics(driver_race_df, driver_vfm_df, driver_affinity_df,
                              driver_char_affinity_df, calendar_df, tracks_df):
    """Process comprehensive statistics for all drivers"""
    driver_stats = []
    race_columns = [col for col in driver_race_df.columns if col.startswith('Race')]

    for _, driver_row in driver_vfm_df.iterrows():
        driver_name = driver_row['Driver']

        # Get race data for this driver
        race_data = driver_race_df[driver_race_df['Driver'] == driver_name]
        if race_data.empty:
            continue

        race_points = race_data[race_columns].values[0]
        valid_points = [float(p) for p in race_points if not np.isnan(p)]

        # Basic statistics
        stats = {
            'name': str(driver_name),
            'team': str(driver_row.get('Team', 'Unknown')),
            'cost': str(driver_row['Cost']),
            'cost_value': float(re.sub(r'[^\d.]', '', str(driver_row['Cost']))),
            'vfm': float(driver_row['VFM']),
            'trend': str(driver_row.get('Performance_Trend', 'Unknown')),
            'avg_points': float(np.mean(valid_points)) if valid_points else 0.0,
            'total_points': float(np.sum(valid_points)) if valid_points else 0.0,
            'races_completed': int(len(valid_points)),
            'consistency': float(np.std(valid_points)) if len(valid_points) > 1 else 0.0,
            'best_race': float(np.max(valid_points)) if valid_points else 0.0,
            'worst_race': float(np.min(valid_points)) if valid_points else 0.0,
        }

        # Recent form
        if len(valid_points) >= 3:
            recent_avg = np.mean(valid_points[-3:])
            stats['recent_form'] = float(recent_avg - stats['avg_points'])
        else:
            stats['recent_form'] = 0.0

        # Track characteristic affinities
        if driver_name in driver_char_affinity_df.index:
            char_affinities = driver_char_affinity_df.loc[driver_name]
            stats['char_affinities'] = {
                'Corners': float(char_affinities.get('Corners', 0)),
                'Length': float(char_affinities.get('Length (km)', 0)),
                'Overtaking': float(char_affinities.get('Overtaking Opportunities_encoded', 0)),
                'Speed': float(char_affinities.get('Track Speed_encoded', 0)),
                'Temperature': float(char_affinities.get('Expected Temperatures_encoded', 0))
            }
        else:
            stats['char_affinities'] = {
                'Corners': 0.0, 'Length': 0.0, 'Overtaking': 0.0, 'Speed': 0.0, 'Temperature': 0.0
            }

        # Track-specific performance
        track_performance = []
        for _, cal_row in calendar_df.iterrows():
            race = cal_row['Race']
            if race in race_columns:
                race_idx = race_columns.index(race)
                if race_idx < len(race_points) and not np.isnan(race_points[race_idx]):
                    circuit = str(cal_row['Circuit'])
                    affinity_col = f'{driver_name}_affinity'

                    circuit_affinity = 0.0
                    if affinity_col in driver_affinity_df.columns:
                        circuit_row = driver_affinity_df[driver_affinity_df['Circuit'] == circuit]
                        if not circuit_row.empty:
                            circuit_affinity = float(circuit_row.iloc[0][affinity_col])

                    track_performance.append({
                        'circuit': circuit,
                        'points': float(race_points[race_idx]),
                        'affinity': circuit_affinity
                    })

        track_performance.sort(key=lambda x: x['points'], reverse=True)
        stats['best_tracks'] = track_performance[:3] if len(track_performance) >= 3 else track_performance
        stats['worst_tracks'] = track_performance[-3:] if len(track_performance) >= 3 else []

        # Upcoming races affinity
        upcoming_races = []
        races_completed = len([r for r in race_columns if r in calendar_df['Race'].values])
        for i in range(races_completed + 1, min(races_completed + 4, len(race_columns) + 1)):
            race_name = f'Race{i}'
            race_info = calendar_df[calendar_df['Race'] == race_name]
            if not race_info.empty:
                circuit = str(race_info.iloc[0]['Circuit'])
                affinity_col = f'{driver_name}_affinity'

                circuit_affinity = 0.0
                if affinity_col in driver_affinity_df.columns:
                    circuit_row = driver_affinity_df[driver_affinity_df['Circuit'] == circuit]
                    if not circuit_row.empty:
                        circuit_affinity = float(circuit_row.iloc[0][affinity_col])

                upcoming_races.append({
                    'race': race_name,
                    'circuit': circuit,
                    'affinity': circuit_affinity
                })

        stats['upcoming_races'] = upcoming_races
        driver_stats.append(stats)

    # Calculate cost efficiency rankings
    driver_stats.sort(key=lambda x: x['vfm'], reverse=True)
    for i, driver in enumerate(driver_stats):
        driver['vfm_rank'] = i + 1

    return driver_stats


def process_constructor_statistics(constructor_race_df, constructor_vfm_df, constructor_affinity_df,
                                   constructor_char_affinity_df, calendar_df, tracks_df):
    """Process comprehensive statistics for all constructors"""
    constructor_stats = []
    race_columns = [col for col in constructor_race_df.columns if col.startswith('Race')]

    for _, constructor_row in constructor_vfm_df.iterrows():
        constructor_name = constructor_row['Constructor']

        # Get race data for this constructor
        race_data = constructor_race_df[constructor_race_df['Constructor'] == constructor_name]
        if race_data.empty:
            continue

        race_points = race_data[race_columns].values[0]
        valid_points = [float(p) for p in race_points if not np.isnan(p)]

        # Basic statistics
        stats = {
            'name': str(constructor_name),
            'cost': str(constructor_row['Cost']),
            'cost_value': float(re.sub(r'[^\d.]', '', str(constructor_row['Cost']))),
            'vfm': float(constructor_row['VFM']),
            'trend': str(constructor_row.get('Performance_Trend', 'Unknown')),
            'avg_points': float(np.mean(valid_points)) if valid_points else 0.0,
            'total_points': float(np.sum(valid_points)) if valid_points else 0.0,
            'races_completed': int(len(valid_points)),
            'consistency': float(np.std(valid_points)) if len(valid_points) > 1 else 0.0,
            'best_race': float(np.max(valid_points)) if valid_points else 0.0,
            'worst_race': float(np.min(valid_points)) if valid_points else 0.0,
        }

        # Recent form
        if len(valid_points) >= 3:
            recent_avg = np.mean(valid_points[-3:])
            stats['recent_form'] = float(recent_avg - stats['avg_points'])
        else:
            stats['recent_form'] = 0.0

        # Track characteristic affinities
        if constructor_name in constructor_char_affinity_df.index:
            char_affinities = constructor_char_affinity_df.loc[constructor_name]
            stats['char_affinities'] = {
                'Corners': float(char_affinities.get('Corners', 0)),
                'Length': float(char_affinities.get('Length (km)', 0)),
                'Overtaking': float(char_affinities.get('Overtaking Opportunities_encoded', 0)),
                'Speed': float(char_affinities.get('Track Speed_encoded', 0)),
                'Temperature': float(char_affinities.get('Expected Temperatures_encoded', 0))
            }
        else:
            stats['char_affinities'] = {
                'Corners': 0.0, 'Length': 0.0, 'Overtaking': 0.0, 'Speed': 0.0, 'Temperature': 0.0
            }

        # Track-specific performance
        track_performance = []
        for _, cal_row in calendar_df.iterrows():
            race = cal_row['Race']
            if race in race_columns:
                race_idx = race_columns.index(race)
                if race_idx < len(race_points) and not np.isnan(race_points[race_idx]):
                    circuit = str(cal_row['Circuit'])
                    affinity_col = f'{constructor_name}_affinity'

                    circuit_affinity = 0.0
                    if affinity_col in constructor_affinity_df.columns:
                        circuit_row = constructor_affinity_df[constructor_affinity_df['Circuit'] == circuit]
                        if not circuit_row.empty:
                            circuit_affinity = float(circuit_row.iloc[0][affinity_col])

                    track_performance.append({
                        'circuit': circuit,
                        'points': float(race_points[race_idx]),
                        'affinity': circuit_affinity
                    })

        track_performance.sort(key=lambda x: x['points'], reverse=True)
        stats['best_tracks'] = track_performance[:3] if len(track_performance) >= 3 else track_performance
        stats['worst_tracks'] = track_performance[-3:] if len(track_performance) >= 3 else []

        # Upcoming races affinity
        upcoming_races = []
        races_completed = len([r for r in race_columns if r in calendar_df['Race'].values])
        for i in range(races_completed + 1, min(races_completed + 4, len(race_columns) + 1)):
            race_name = f'Race{i}'
            race_info = calendar_df[calendar_df['Race'] == race_name]
            if not race_info.empty:
                circuit = str(race_info.iloc[0]['Circuit'])
                affinity_col = f'{constructor_name}_affinity'

                circuit_affinity = 0.0
                if affinity_col in constructor_affinity_df.columns:
                    circuit_row = constructor_affinity_df[constructor_affinity_df['Circuit'] == circuit]
                    if not circuit_row.empty:
                        circuit_affinity = float(circuit_row.iloc[0][affinity_col])

                upcoming_races.append({
                    'race': race_name,
                    'circuit': circuit,
                    'affinity': circuit_affinity
                })

        stats['upcoming_races'] = upcoming_races
        constructor_stats.append(stats)

    # Calculate rankings
    constructor_stats.sort(key=lambda x: x['vfm'], reverse=True)
    for i, constructor in enumerate(constructor_stats):
        constructor['vfm_rank'] = i + 1

    return constructor_stats


@app.route('/api/export_statistics')
def export_statistics():
    """Export all statistics as CSV"""
    try:
        stats_response = get_statistics()
        stats_data = stats_response.get_json()

        if not stats_data['success']:
            return stats_response

        # Create CSV for drivers
        driver_rows = []
        for driver in stats_data['drivers']:
            row = {
                'Driver': driver['name'],
                'Team': driver['team'],
                'Cost': driver['cost'],
                'VFM': driver['vfm'],
                'VFM_Rank': driver['vfm_rank'],
                'Avg_Points': round(driver['avg_points'], 2),
                'Total_Points': driver['total_points'],
                'Consistency': round(driver['consistency'], 2),
                'Recent_Form': round(driver['recent_form'], 2),
                'Trend': driver['trend']
            }
            driver_rows.append(row)

        driver_df = pd.DataFrame(driver_rows)

        # Create CSV for constructors
        constructor_rows = []
        for constructor in stats_data['constructors']:
            row = {
                'Constructor': constructor['name'],
                'Cost': constructor['cost'],
                'VFM': constructor['vfm'],
                'VFM_Rank': constructor['vfm_rank'],
                'Avg_Points': round(constructor['avg_points'], 2),
                'Total_Points': constructor['total_points'],
                'Consistency': round(constructor['consistency'], 2),
                'Recent_Form': round(constructor['recent_form'], 2),
                'Trend': constructor['trend']
            }
            constructor_rows.append(row)

        constructor_df = pd.DataFrame(constructor_rows)

        # Save to temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(app.config['RESULTS_FOLDER'], f'statistics_{timestamp}.xlsx')

        with pd.ExcelWriter(output_file) as writer:
            driver_df.to_excel(writer, sheet_name='Drivers', index=False)
            constructor_df.to_excel(writer, sheet_name='Constructors', index=False)

        return send_file(output_file, as_attachment=True,
                         download_name=f'f1_statistics_{timestamp}.xlsx')

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error exporting statistics: {str(e)}'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
