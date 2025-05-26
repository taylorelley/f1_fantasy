# app.py
import os
import json
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import re
from werkzeug.utils import secure_filename
from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer, get_races_completed

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store session data (in production, use proper session management)
sessions = {}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        session_id = request.form.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
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
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_folder, filename)
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
        races_completed = get_races_completed(session_folder + '/')
        
        # Read drivers and constructors for selection
        driver_df = pd.read_csv(os.path.join(session_folder, 'driver_race_data.csv'))
        constructor_df = pd.read_csv(os.path.join(session_folder, 'constructor_race_data.csv'))
        
        drivers_list = sorted(driver_df['Driver'].unique().tolist())
        constructors_list = sorted(constructor_df['Constructor'].unique().tolist())
        
        # Store session info
        sessions[session_id] = {
            'folder': session_folder,
            'races_completed': races_completed,
            'drivers': drivers_list,
            'constructors': constructors_list
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'races_completed': races_completed,
            'drivers': drivers_list,
            'constructors': constructors_list
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
        session_id = data['session_id']
        
        if session_id not in sessions:
            return jsonify({
                'success': False,
                'message': 'Session not found. Please upload files first.'
            })
        
        # Build configuration
        config = {
            'base_path': sessions[session_id]['folder'] + '/',
            'races_completed': sessions[session_id]['races_completed'],
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
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'optimization_{session_id}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify(results)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error during optimization: {str(e)}'
        })

@app.route('/download/<session_id>')
def download_results(session_id):
    """Download optimization results"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f'optimization_{session_id}.json')
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True, download_name=f'f1_optimization_{session_id}.json')
    else:
        return jsonify({'error': 'Results not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
