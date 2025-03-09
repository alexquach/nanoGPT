"""
Visualization tool for MLP neuron activations.
Displays samples with the highest activations for selected neurons.
"""
import os
import argparse
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for
from collections import defaultdict
import json
import math

# Create a custom JSON encoder to handle NumPy types and NaN values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            # Handle NaN values
            if np.isnan(obj):
                return None  # Convert NaN to None, which becomes null in JSON
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__, template_folder='templates', static_folder='static')
# Configure Flask to use our custom JSON encoder
app.json.encoder = NumpyEncoder

# Global variables to store data for multiple models
models = {
    'base': {
        'name': 'BaseGPT',
        'file': 'out_base/mlp_activations_base_posttopk_0.0013020833333333333.csv',
        'df': None,
        'neuron_columns': [],
        'samples_data': {},
        'loaded': False
    },
    'sparse': {
        'name': 'SparseGPT',
        'file': 'out/mlp_activations__posttopk_0.0013020833333333333.csv',
        'df': None,
        'neuron_columns': [],
        'samples_data': {},
        'loaded': False
    },
    'sparse8x': {
        'name': 'Sparse8xGPT',
        'file': 'out_8x_sparse/mlp_activations_8x_sparse_posttopk_0.00016276041666666666.csv',
        'df': None,
        'neuron_columns': [],
        'samples_data': {},
        'loaded': False
    }
}

# Currently active model
active_model = 'base'

def safe_convert(value, default=""):
    """Safely convert a value, handling NaN and None"""
    if pd.isna(value) or value is None:
        return default
    return value

def load_data(model_key):
    """Load and process the activations CSV file for a specific model"""
    global models
    
    model = models[model_key]
    csv_path = model['file']
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return False
    
    print(f"Loading data for {model['name']} from {csv_path}...")
    model['df'] = pd.read_csv(csv_path)
    
    # Extract neuron columns (those starting with 'neuron_')
    model['neuron_columns'] = [col for col in model['df'].columns if col.startswith('neuron_')]
    print(f"Found {len(model['neuron_columns'])} neurons in {model['name']}")
    
    # Add debug output to track neuron columns for each model
    if len(model['neuron_columns']) > 0:
        print(f"First few neurons in {model['name']}: {model['neuron_columns'][:5]}")
    
    # Organize data by sample
    model['samples_data'] = {}
    
    # Process all samples
    for sample_idx in model['df']['sample_idx'].unique():
        sample_df = model['df'][model['df']['sample_idx'] == sample_idx]
        
        # Get the full prompt for this sample (the longest prompt_prefix)
        try:
            full_prompt = sample_df.loc[sample_df['position'].idxmax(), 'prompt_prefix']
            if pd.isna(full_prompt):
                full_prompt = "No prompt available"
        except:
            full_prompt = "Error retrieving prompt"
        
        # Create token-level data for this sample
        tokens = []
        for _, row in sample_df.iterrows():
            position = row['position']
            token = safe_convert(row['token'])
            
            # Get activations for all neurons at this position
            # Convert any NumPy types to Python native types and handle NaN
            activations = {}
            for col in model['neuron_columns']:
                if pd.isna(row[col]):
                    activations[col] = 0.0
                else:
                    val = float(row[col])
                    # Handle potential NaN after conversion
                    activations[col] = 0.0 if math.isnan(val) else val
            
            tokens.append({
                'position': int(position) if isinstance(position, np.integer) else position,
                'token': token,
                'activations': activations
            })
        
        # Store sample data - convert sample_idx to standard Python int
        sample_idx_py = int(sample_idx) if isinstance(sample_idx, np.integer) else sample_idx
        model['samples_data'][sample_idx_py] = {
            'prompt': full_prompt,
            'tokens': tokens
        }
    
    print(f"Processed {len(model['samples_data'])} samples for {model['name']}")
    model['loaded'] = True
    return True

def get_top_samples_for_neuron(model_key, neuron_id, top_n=10):
    """Find samples with highest activations for a given neuron in the specified model"""
    model = models[model_key]
    
    # Calculate max activation per sample for this neuron
    sample_max_activations = {}
    
    for sample_idx, sample in model['samples_data'].items():
        # Get max activation across all tokens in this sample
        max_activation = 0.0
        for token in sample['tokens']:
            activation = token['activations'].get(neuron_id, 0.0)
            if not math.isnan(activation):  # Skip NaN values when finding max
                max_activation = max(max_activation, activation)
        
        sample_max_activations[sample_idx] = max_activation
    
    # Sort samples by max activation
    sorted_samples = sorted(sample_max_activations.items(), 
                           key=lambda x: x[1], reverse=True)
    
    # Return top N samples
    top_samples = []
    for sample_idx, max_activation in sorted_samples[:top_n]:
        # For each token, get its activation for this neuron
        tokens_with_activations = []
        for token in model['samples_data'][sample_idx]['tokens']:
            activation = token['activations'].get(neuron_id, 0.0)
            # Handle NaN in activations
            if math.isnan(activation):
                activation = 0.0
                
            tokens_with_activations.append({
                'token': token['token'],
                'activation': float(activation)  # Ensure activation is a Python float
            })
        
        top_samples.append({
            'sample_idx': int(sample_idx) if isinstance(sample_idx, np.integer) else sample_idx,
            'prompt': model['samples_data'][sample_idx]['prompt'],
            'max_activation': float(max_activation),  # Ensure max_activation is a Python float
            'tokens': tokens_with_activations
        })
    
    return top_samples

def get_neuron_status_for_model(model_key, threshold=0.1):
    """Get activation status for all neurons in the specified model"""
    model = models[model_key]
    status = {}
    
    for neuron_id in model['neuron_columns']:
        # Check if this neuron has any significant activations
        has_activations = False
        max_activation = 0.0
        
        # Check all samples for this neuron to find the true global maximum
        for sample_idx, sample in model['samples_data'].items():
            for token in sample['tokens']:
                activation = token['activations'].get(neuron_id, 0.0)
                if not math.isnan(activation):
                    if activation > threshold:
                        has_activations = True
                    max_activation = max(max_activation, activation)
        
        status[neuron_id] = {
            'has_activations': has_activations,
            'max_activation': float(max_activation)
        }
    
    return status

@app.route('/')
def index():
    """Render the main visualization page"""
    global active_model
    
    # Get the model parameter if provided, otherwise use the active model
    model_key = request.args.get('model', active_model)
    
    # Ensure the requested model is valid
    if model_key not in models:
        model_key = 'base'
    
    # Set the active model
    active_model = model_key
    
    # All models should already be loaded, no need for conditional loading
    
    return render_template('index.html', 
                          neurons=models[active_model]['neuron_columns'],
                          models=models,
                          active_model=active_model)

@app.route('/api/switch_model/<model_key>')
def switch_model(model_key):
    """Switch to the specified model"""
    global active_model
    
    if model_key in models:
        # Ensure model data is loaded
        if not models[model_key]['loaded']:
            load_data(model_key)
            
        active_model = model_key
        
        # Debug output to verify correct neuron list is being sent
        print(f"Switching to model: {model_key}")
        print(f"Number of neurons for {model_key}: {len(models[model_key]['neuron_columns'])}")
        if len(models[model_key]['neuron_columns']) > 0:
            print(f"First few neurons: {models[model_key]['neuron_columns'][:5]}")
        
        return jsonify({
            "success": True,
            "model": models[active_model]['name'],
            "neurons": models[active_model]['neuron_columns']
        })
    else:
        return jsonify({"error": "Invalid model key"}), 400

@app.route('/api/neurons')
def get_neurons():
    """Return list of all neuron identifiers for the active model"""
    global active_model
    return jsonify(models[active_model]['neuron_columns'])

@app.route('/api/neuron/<neuron_id>')
def get_neuron_data(neuron_id):
    """Return top samples for a specific neuron in the active model"""
    global active_model
    
    try:
        top_n = int(request.args.get('top_n', 10))
        top_samples = get_top_samples_for_neuron(active_model, neuron_id, top_n)
        return jsonify(top_samples)
    except Exception as e:
        print(f"Error processing request for neuron {neuron_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/neuron_status')
def get_neuron_status():
    """Return activation status for all neurons in the active model"""
    global active_model
    
    try:
        threshold = float(request.args.get('threshold', 0.1))
        status = get_neuron_status_for_model(active_model, threshold)
        return jsonify(status)
    except Exception as e:
        print(f"Error getting neuron status: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description='Visualize neuron activations')
    parser.add_argument('--base_csv', type=str, 
                       default='out_base/mlp_activations_base_posttopk_0.0013020833333333333.csv',
                       help='Path to the BaseGPT activations CSV file')
    parser.add_argument('--sparse_csv', type=str, 
                       default='out/mlp_activations__posttopk_0.0013020833333333333.csv',
                       help='Path to the SparseGPT activations CSV file')
    parser.add_argument('--sparse8x_csv', type=str, 
                       default='out_8x_sparse/mlp_activations_8x_sparse_posttopk_0.00016276041666666666.csv',
                       help='Path to the Sparse8xGPT activations CSV file')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')
    parser.add_argument('--start_model', type=str, choices=['base', 'sparse', 'sparse8x'], default='base',
                       help='Which model to load initially')
    args = parser.parse_args()
    
    # Update file paths
    models['base']['file'] = args.base_csv
    models['sparse']['file'] = args.sparse_csv
    models['sparse8x']['file'] = args.sparse8x_csv
    
    # Set initial active model
    global active_model
    active_model = args.start_model
    
    # Pre-load both models at startup
    print("Loading all model data at startup...")
    for model_key in models:
        success = load_data(model_key)
        if not success:
            print(f"WARNING: Failed to load data for {model_key} model")
    
    # Verify each model has unique neuron columns
    print("\nVerifying neuron columns for each model:")
    for model_key, model_data in models.items():
        print(f"{model_key}: {len(model_data['neuron_columns'])} neurons")
        if len(model_data['neuron_columns']) > 0:
            print(f"Sample neurons: {model_data['neuron_columns'][:3]}")
    
    # Run the Flask app
    print(f"Starting visualization server on http://localhost:{args.port}")
    app.run(host='0.0.0.0', debug=True, port=args.port)

if __name__ == '__main__':
    main() 