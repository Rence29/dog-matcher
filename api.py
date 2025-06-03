from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import re # For cleaning Dog_Age input

app = Flask(__name__)

# --- Configuration ---
MODEL_FILE = 'random_forest_dog_matcher_model.pkl'
ENCODED_FEATURES_FILE = 'encoded_features.txt'
PORT = 5000 # Choose an available port (make sure it's not in use by other applications)

# --- Load the Trained Model and Feature List ---
# These are loaded only once when the API starts, for efficiency.
try:
    model = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")

    with open(ENCODED_FEATURES_FILE, 'r') as f:
        expected_features = [line.strip() for line in f]
    print(f"Loaded {len(expected_features)} expected features from {ENCODED_FEATURES_FILE}.")

except FileNotFoundError:
    print(f"Error: Required file not found. Make sure '{MODEL_FILE}' and '{ENCODED_FEATURES_FILE}' exist.")
    print("Please run preprocess_data.py and train_model.py first.")
    exit()
except Exception as e:
    print(f"Error loading model or features: {e}")
    exit()

# Function to preprocess incoming data from the API request
def preprocess_input(data):
    """
    Applies the same preprocessing steps to a single input data point
    as were applied to the training data.
    """
    processed_data = {}

    # Dog_Age_Years
    dog_age_str = data.get('Dog_Age')
    if dog_age_str:
        try:
            # Handle formats like "2 years", "1.5 years"
            if 'years' in dog_age_str:
                processed_data['Dog_Age_Years'] = float(dog_age_str.replace(' years', '').replace(' year', ''))
            elif 'months' in dog_age_str:
                processed_data['Dog_Age_Years'] = float(dog_age_str.replace(' months', '').replace(' month', '')) / 12
            else: # Assume it's already a number
                 processed_data['Dog_Age_Years'] = float(dog_age_str)
        except ValueError:
            processed_data['Dog_Age_Years'] = None # Handle bad age input
    else:
        processed_data['Dog_Age_Years'] = None # Handle missing age input

    # One-hot encode other categorical features
    categorical_cols = [
        'Breed', 'Behavior', 'Size', 'Health_Condition',
        'Housing_Type', 'Lifestyle', 'Family_Composition', 'Pet_Experience'
    ]

    # Initialize all possible encoded feature columns to 0
    # This is critical to ensure the input DataFrame has the same columns as training data
    for feature_name in expected_features:
        if feature_name not in processed_data: # Don't overwrite Dog_Age_Years
            processed_data[feature_name] = 0

    for col in categorical_cols:
        value = data.get(col)
        if value:
            # Construct the one-hot encoded column name as it was in training
            # e.g., 'Behavior_Energetic' from 'Behavior' and 'Energetic'
            encoded_col_name = f"{col}_{value.replace(' ', '_')}" # Replace spaces for consistency if values have spaces
            if encoded_col_name in expected_features:
                processed_data[encoded_col_name] = 1

    # Create a DataFrame from the processed data, ensuring correct column order
    # Any features not present in the input will default to 0 as initialized above
    input_df = pd.DataFrame([processed_data], columns=expected_features)

    return input_df

@app.route('/predict_match', methods=['POST'])
def predict_match():
    """
    Receives adopter and dog data (combined in one JSON),
    preprocesses it, makes a prediction using the loaded model,
    and returns the prediction and probabilities.
    """
    if not request.json:
        return jsonify({'error': 'Request must be JSON'}), 400

    raw_data = request.json
    print(f"Received raw prediction request data: {raw_data}")

    # Preprocess the incoming data
    try:
        input_for_prediction = preprocess_input(raw_data)
        # print("Preprocessed input for prediction:")
        # print(input_for_prediction) # Uncomment for debugging

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return jsonify({'error': f'Error during input preprocessing: {e}'}), 400

    # Ensure the preprocessed DataFrame has the same number of columns and order as expected
    if list(input_for_prediction.columns) != expected_features:
        # This check is crucial for robustness
        print(f"Mismatched features. Expected: {expected_features}, Got: {list(input_for_prediction.columns)}")
        # You might want to log the difference for debugging
        return jsonify({'error': 'Input features do not match expected model features. Check your data.'}), 400

    try:
        # Make prediction and get probabilities
        prediction = model.predict(input_for_prediction)[0] # Get the first (and only) prediction
        prediction_proba = model.predict_proba(input_for_prediction)[0] # Probabilities for [class 0, class 1]

        # Return the results
        return jsonify({
            'match_prediction': int(prediction), # 0 or 1
            'probability_bad_match': float(prediction_proba[0]),
            'probability_good_match': float(prediction_proba[1]),
            'message': 'Prediction successful'
        }), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to confirm API is running."""
    status = 'healthy'
    model_loaded = False
    if 'model' in globals() and model is not None:
        model_loaded = True
    else:
        status = 'degraded - model not loaded'
    return jsonify({'status': status, 'model_loaded': model_loaded, 'port': PORT}), 200


if __name__ == '__main__':
    print(f"Starting Flask API on http://127.0.0.1:{PORT}")
    print("Keep this terminal open for the API to run.")
    app.run(debug=True, port=PORT, host='0.0.0.0') # host='0.0.0.0' makes it accessible from other machines on the network (useful for testing)