import os
import subprocess
import sys
import time
import pandas as pd
import numpy as np
import requests
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Configure Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load Environment Variables
load_dotenv()

# Initialize Flask App 
app = Flask(__name__)
CORS(app)

#Configuration & Startup Validation
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
KAGGLE_DATASET_SLUG = os.getenv("KAGGLE_DATASET_SLUG")
KAGGLE_NOTEBOOK_SLUG = os.getenv("KAGGLE_NOTEBOOK_SLUG")
KAGGLE_MODELS_DATASET_SLUG = os.getenv("KAGGLE_MODELS_DATASET_SLUG")

# Use environment variable for Kaggle path, but fall back to a common location
KAGGLE_EXECUTABLE_PATH = os.getenv("KAGGLE_EXECUTABLE_PATH")
if not KAGGLE_EXECUTABLE_PATH:
    python_dir = os.path.dirname(sys.executable)
    KAGGLE_EXECUTABLE_PATH = os.path.join(python_dir, 'Scripts', 'kaggle.exe')

# Validate configuration on startup
if not ETHERSCAN_API_KEY:
    logging.critical("FATAL ERROR: ETHERSCAN_API_KEY is not set.")
if not KAGGLE_DATASET_SLUG:
    logging.critical("FATAL ERROR: KAGGLE_DATASET_SLUG is not set.")
if not KAGGLE_NOTEBOOK_SLUG:
    logging.critical("FATAL ERROR: KAGGLE_NOTEBOOK_SLUG is not set.")
if not KAGGLE_MODELS_DATASET_SLUG:
    logging.critical("FATAL ERROR: KAGGLE_MODELS_DATASET_SLUG is not set.")
if not os.path.exists(KAGGLE_EXECUTABLE_PATH):
    logging.critical(f"FATAL ERROR: Kaggle executable not found at '{KAGGLE_EXECUTABLE_PATH}'.")

# Helper Function for Kaggle Commands
def run_kaggle_command(command_args, capture_output=False):
    """
    A robust helper function to run Kaggle CLI commands.
    Suppresses output when not needed to avoid encoding errors.
    """
    command = [KAGGLE_EXECUTABLE_PATH] + command_args
    logging.info(f"Executing command: {' '.join(command)}")
    
    try:
        if not capture_output:
            # For commands where we don't need to read the output
            subprocess.run(
                command, check=True, timeout=120, 
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return None # Explicitly return None for success
        else:
            # For commands where we MUST read the output (like status checks)
            result = subprocess.run(
                command, check=True, timeout=120, capture_output=True, 
                text=True, encoding='utf-8', errors='replace'
            )
            return result.stdout.strip()
            
    except subprocess.CalledProcessError as e:
        # This exception is raised if the command returns a non-zero exit code
        logging.error("A Kaggle CLI command returned a non-zero exit code.")
        logging.error(f"Failed Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        # Try to decode stdout/stderr for better logging if they were captured
        stdout_decoded = e.stdout.decode('utf-8', errors='replace') if e.stdout else 'N/A'
        stderr_decoded = e.stderr.decode('utf-8', errors='replace') if e.stderr else 'N/A'
        logging.error(f"STDOUT: {stdout_decoded}")
        logging.error(f"STDERR: {stderr_decoded}")
        raise  # Re-raise the exception to be handled by the main endpoint logic

# Feature Engineering Function (unchanged)
def create_feature_row(tx_list, address):
    if not tx_list: return None
    df = pd.DataFrame(tx_list)
    numeric_cols = ['value', 'gas', 'gasPrice', 'gasUsed', 'timeStamp', 'isError']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['value_eth'] = df['value'] / 1e18
    df['tx_fee_eth'] = (df['gasUsed'] * df['gasPrice']) / 1e18
    address_lower = address.lower()
    df['from_lower'] = df['from'].str.lower()
    df['to_lower'] = df['to'].str.lower()
    sent_tx = df[df['from_lower'] == address_lower]
    received_tx = df[df['to_lower'] == address_lower]
    df_sorted = df.sort_values(by='timeStamp').reset_index(drop=True)
    time_diffs = df_sorted['timeStamp'].diff().dropna()
    sent_tx_sorted = sent_tx.sort_values(by='timeStamp').reset_index(drop=True)
    sent_time_diffs = sent_tx_sorted['timeStamp'].diff().dropna()
    features = {
        'ADDRESS': address, 'total_tx_count': len(df), 'avg_gas_used': df['gasUsed'].mean(),
        'unique_tx_partners_out': sent_tx['to_lower'].nunique(), 'unique_tx_partners_in': received_tx['from_lower'].nunique(),
        'total_value_sent': sent_tx['value_eth'].sum(), 'avg_tx_value_sent': sent_tx['value_eth'].mean(),
        'max_tx_value_sent': sent_tx['value_eth'].max(), 'total_tx_fee': df['tx_fee_eth'].sum(),
        'avg_tx_fee': df['tx_fee_eth'].mean(), 'max_tx_fee': df['tx_fee_eth'].max(),
        'num_tx_sent': len(sent_tx), 'total_value_received': received_tx['value_eth'].sum(),
        'avg_tx_value_received': received_tx['value_eth'].mean(), 'max_tx_value_received': received_tx['value_eth'].max(),
        'num_tx_received': len(received_tx), 'avg_time_between_tx_overall': time_diffs.mean(),
        'min_time_between_tx_overall': time_diffs.min(), 'max_time_between_tx_overall': time_diffs.max(),
        'avg_time_between_outgoing_tx': sent_time_diffs.mean(),
        'activity_duration_seconds': df['timeStamp'].max() - df['timeStamp'].min() if not df.empty else 0,
        'ratio_sent_received_value': (sent_tx['value_eth'].sum() / received_tx['value_eth'].sum()) if received_tx['value_eth'].sum() > 0 else 0,
        'ratio_sent_received_count': (len(sent_tx) / len(received_tx)) if len(received_tx) > 0 else 0,
        'total_activity_types': len(df['type'].unique()) if 'type' in df.columns else (1 if len(sent_tx)>0 or len(received_tx)>0 else 0),
        'total_interactions': len(df),
    }
    feature_df = pd.DataFrame([features])
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0, inplace=True)
    return feature_df

# Notebook Creation Function (unchanged) 
def create_notebook_file(path, title, data_slug, models_slug):
    code_cell_source = f"""
    import pandas as pd; import numpy as np; import joblib; import warnings
warnings.filterwarnings('ignore')
DATA_FILE_PATH = "/kaggle/input/{data_slug}/features.csv"
MODEL_FILE_PATH = "/kaggle/input/{models_slug}/artifacts/xgboost_model.pkl"
TRANSFORMER_FILE_PATH = "/kaggle/input/{models_slug}/artifacts/power_transformer.pkl"
OUTPUT_FILE_PATH = "/kaggle/working/prediction_output.csv"
print("--- Loading Files ---")
try:
    new_data_df = pd.read_csv(DATA_FILE_PATH)
    print(f"✅ Loaded data for address: {{new_data_df['ADDRESS'].iloc[0]}}")
except Exception as e: print(f"❌ Error loading data file: {{e}}"); exit()
try:
    model = joblib.load(MODEL_FILE_PATH)
    print("✅ Loaded XGBoost model.")
except Exception as e: print(f"❌ Error loading model file: {{e}}"); exit()
try:
    pt = joblib.load(TRANSFORMER_FILE_PATH)
    print("✅ Loaded PowerTransformer.")
except Exception as e: print(f"❌ Error loading transformer file: {{e}}"); exit()
print("\\n--- Preprocessing New Data ---")
transformer_columns = pt.feature_names_in_
X_new_trans = pd.DataFrame(columns=transformer_columns)
for col in transformer_columns: X_new_trans[col] = new_data_df[col] if col in new_data_df.columns else 0
print("Applying Yeo-Johnson transformation...")
X_transformed = pt.transform(X_new_trans)
X_transformed_df = pd.DataFrame(X_transformed, columns=transformer_columns)
training_columns = model.feature_names_in_
X_final = pd.DataFrame(columns=training_columns)
for col in training_columns: X_final[col] = X_transformed_df[col] if col in X_transformed_df.columns else 0
print("✅ Data transformed and aligned.")
print("\\n--- Making Prediction ---")
prediction = model.predict(X_final)
predicted_label = int(prediction[0])
prediction_proba = model.predict_proba(X_final)
confidence_score = float(prediction_proba[0].max())
print(f"Predicted Label: {{predicted_label}}")
print(f"Confidence Score: {{confidence_score:.2%}}")
print("\\n--- Saving Output ---")
result_dict = {{'address': new_data_df['ADDRESS'].iloc[0], 'prediction': predicted_label, 'confidence': confidence_score}}
result_df = pd.DataFrame([result_dict])
result_df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"✅ Prediction saved to {{OUTPUT_FILE_PATH}}")
"""
    notebook_json = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_cell_source.strip()}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.12"}}, "nbformat": 4, "nbformat_minor": 4}
    with open(os.path.join(path, f"{title}.ipynb"), 'w') as f: json.dump(notebook_json, f)

# Main API Endpoint 
@app.route('/analyze', methods=['POST'])
def analyze_address():
    data = request.get_json()
    if not data or not data.get('address'):
        logging.warning("Request failed: No address provided.")
        return jsonify({"error": "Address not provided"}), 400
    
    address = data['address']
    logging.info(f"--- New analysis request received for address: {address} ---")

    #Part A: Fetch and Process Data 
    try:
        logging.info("Step 1: Fetching transaction data from Etherscan...")
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        tx_data = response.json().get('result', [])
        
        if not isinstance(tx_data, list) or not tx_data:
            logging.warning("Etherscan returned no transactions.")
            return jsonify({"error": "No transactions found or Etherscan API returned an error message."}), 404
        
        logging.info(f"Successfully fetched {len(tx_data)} transactions.")
        logging.info("Step 2: Processing raw data into features...")
        feature_df = create_feature_row(tx_data, address)
        if feature_df is None:
            logging.error("Feature creation failed.")
            return jsonify({"error": "Could not process transactions."}), 500
        logging.info("Feature processing complete.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Etherscan API request failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch data from Etherscan."}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during data processing."}), 500

    #Part B: Prepare Local Files for Kaggle
    temp_dir = "temp_data"
    notebook_push_dir = "kaggle_notebook_push"
    try:
        logging.info("Step 3: Preparing local files for Kaggle upload...")
        os.makedirs(temp_dir, exist_ok=True)
        feature_df.to_csv(os.path.join(temp_dir, "features.csv"), index=False)
        with open(os.path.join(temp_dir, "dataset-metadata.json"), 'w') as f:
            json.dump({"title": "Real-time ETH Address Features", "id": KAGGLE_DATASET_SLUG}, f)
        
        os.makedirs(notebook_push_dir, exist_ok=True)
        notebook_title = KAGGLE_NOTEBOOK_SLUG.split('/')[-1]
        kernel_metadata = {
            "id": KAGGLE_NOTEBOOK_SLUG, "title": notebook_title,
            "code_file": f"{notebook_title}.ipynb", "language": "python",
            "kernel_type": "notebook",
            "dataset_sources": [KAGGLE_DATASET_SLUG, KAGGLE_MODELS_DATASET_SLUG]
        }
        with open(os.path.join(notebook_push_dir, "kernel-metadata.json"), 'w') as f:
            json.dump(kernel_metadata, f)
        
        create_notebook_file(notebook_push_dir, notebook_title, KAGGLE_DATASET_SLUG.split('/')[-1], KAGGLE_MODELS_DATASET_SLUG.split('/')[-1])
        logging.info("Local file preparation complete.")
    except Exception as e:
        logging.error(f"Failed to prepare local files: {e}", exc_info=True)
        return jsonify({"error": "Failed to prepare local files for Kaggle."}), 500

    #Part C: Interact with Kaggle API 
    try:
        # 1. Push Dataset & Poll
        logging.info("Step 4a: Pushing new data to Kaggle Dataset...")
        run_kaggle_command(["datasets", "version", "-p", temp_dir, "-m", f"Feature update for {address}"])
        
        logging.info("Step 4b: Waiting for dataset to be processed...")
        start_time = time.time()
        while time.time() - start_time < 120: # 2 minute timeout
            status = run_kaggle_command(["datasets", "status", KAGGLE_DATASET_SLUG], capture_output=True)
            if "ready" in status: break
            time.sleep(10)
        else:
            logging.error("Dataset processing timed out.")
            return jsonify({"error": "Kaggle took too long to process the new data."}), 504
        
        # 2. Push Kernel & Poll
        logging.info("Step 4c: Triggering Kaggle Notebook run...")
        run_kaggle_command(["kernels", "push", "-p", notebook_push_dir])
        
        logging.info("Step 4d: Polling for notebook completion...")
        start_time = time.time()
        while time.time() - start_time < 300: # 5 minute timeout
            status = run_kaggle_command(["kernels", "status", KAGGLE_NOTEBOOK_SLUG], capture_output=True)
            if "complete" in status.lower(): break
            if "error" in status.lower() or "cancelled" in status.lower():
                raise Exception(f"Kaggle notebook run failed with status: {status}")
            time.sleep(20)
        else:
            logging.error("Notebook run timed out.")
            return jsonify({"error": "Analysis timed out. Kaggle notebook took too long."}), 504

        # 3. Fetch Output
        logging.info("Step 4e: Fetching prediction results from Kaggle...")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            run_kaggle_command(["kernels", "output", KAGGLE_NOTEBOOK_SLUG, "-p", output_dir, "--force"])
        except: 
            result_df = pd.read_csv(os.path.join(output_dir, "prediction_output.csv"))
            prediction_result = result_df.to_dict('records')[0]
        
        logging.info(f"--- Analysis successful. Returning result: {prediction_result} ---")
        return jsonify(prediction_result)

    except subprocess.CalledProcessError:
        
        return jsonify({"error": "A Kaggle command failed. Check server logs for details."}), 500
    except FileNotFoundError:
        logging.error("Could not find 'prediction_output.csv'. The notebook may have failed to produce an output.", exc_info=True)
        return jsonify({"error": "Could not find prediction result file from Kaggle."}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during Kaggle interaction: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during Kaggle interaction."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
