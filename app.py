from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)

CORS(app)
with open('adaboost_model.pkl', 'rb') as f:
    adaboost_model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    
    required_columns = ['age', 'blood_pressure', 'blood_glucose_random', 'haemoglobin',
       'hypertension', 'diabetes_mellitus', 'appetite']
    
    for column in required_columns:
        if column not in input_data:
            return jsonify({'error': f'Missing column: {column}'}), 400
    
    # Create a copy of input_data to avoid modifying the original data
    modified_input_data = input_data.copy()
    
    # Convert 1 to "Yes" and 0 to "No" for specific columns
    for column in ['hypertension', 'diabetes_mellitus', 'appetite']:
        if column in modified_input_data:
            modified_input_data[column] = 1 if modified_input_data[column] == "Yes" else 0
    
    input_df = pd.DataFrame([modified_input_data])
    
    predictions = adaboost_model.predict(input_df)
    
    return jsonify({'predictions': predictions.tolist()}), 200



if __name__ == '__main__':
    app.run(debug=True)