# app.py
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
from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer, get_races_completed

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
            'constructors': constructors_list
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
            'updated_default': update_default
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error uploading files: {str(e)}'
        })

@app.route('/optimize', methods=['POST'])
def optimize():
    """Run the optimization"""
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
            'use_parallel': False  # Disable for web app
        }
        
        # Save configuration for easy reuse
        config_file = os.path.join(app.config['RESULTS_FOLDER'], 'last_config.json')
        with open(config_file, 'w') as f:
            json.dump({
                'current_drivers': data['current_drivers'],
                'current_constructors': data['current_constructors'],
                'remaining_budget': data['remaining_budget'],
                'step1_swaps': data['step1_swaps'],
                'step2_swaps': data['step2_swaps'],
                'weighting_scheme': data['weighting_scheme'],
                'risk_tolerance': data['risk_tolerance'],
                'multiplier': data['multiplier']
            }, f)
        
        results = {
            'status': 'running',
            'progress': []
        }
        
        # Step 1: Calculate VFM
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
        
        # Get VFM data for display
        results['vfm_data'] = {
            'drivers': driver_vfm.head(10).to_dict('records'),
            'constructors': constructor_vfm.to_dict('records')
        }
        
        results['status'] = 'complete'
        results['success'] = True
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'optimization_{timestamp}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['result_id'] = timestamp
        
        return jsonify(results)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error during optimization: {str(e)}'
        })

@app.route('/download/<result_id>')
def download_results(result_id):
    """Download optimization results"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f'optimization_{result_id}.json')
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True, download_name=f'f1_optimization_{result_id}.json')
    else:
        return jsonify({'error': 'Results not found'}), 404

@app.route('/load_config')
def load_config():
    """Load last used configuration"""
    config_file = os.path.join(app.config['RESULTS_FOLDER'], 'last_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return jsonify({'success': True, 'config': config})
    return jsonify({'success': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
