from flask import Flask, request, jsonify, render_template
import pandas as pd 
import numpy as np 
import pickle as pk 
import os

app = Flask(__name__)

# Load model
try:
    model = pk.load(open('model.pkl', 'rb'))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Load and process car data
try:
    cars_data = pd.read_csv('Cardetails.csv')
    
    def get_brand_name(car_name):
        car_name = car_name.split(' ')[0]
        return car_name.strip()
    
    cars_data['name'] = cars_data['name'].apply(get_brand_name)
    
    # Get unique values for dropdowns
    BRANDS = sorted(cars_data['name'].unique().tolist())
    FUEL_TYPES = sorted(cars_data['fuel'].unique().tolist())
    SELLER_TYPES = sorted(cars_data['seller_type'].unique().tolist())
    TRANSMISSION_TYPES = sorted(cars_data['transmission'].unique().tolist())
    OWNER_TYPES = sorted(cars_data['owner'].unique().tolist())
    
    print("✓ Data loaded successfully")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    # Default values if CSV not found
    BRANDS = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Mercedes-Benz']
    FUEL_TYPES = ['Petrol', 'Diesel', 'CNG', 'LPG']
    SELLER_TYPES = ['Individual', 'Dealer', 'Trustmark Dealer']
    TRANSMISSION_TYPES = ['Manual', 'Automatic']
    OWNER_TYPES = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']

# Mapping dictionaries for encoding
OWNER_MAPPING = {
    'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
    'Fourth & Above Owner': 4, 'Test Drive Car': 5
}

FUEL_MAPPING = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}

SELLER_MAPPING = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}

TRANSMISSION_MAPPING = {'Manual': 1, 'Automatic': 2}

BRAND_MAPPING = {
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
    'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
    'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14,
    'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
    'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24,
    'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29,
    'Isuzu': 30, 'Opel': 31
}

@app.route('/')
def home():
    """Render the home page with the form"""
    return render_template('index.html', 
                         brands=BRANDS,
                         fuel_types=FUEL_TYPES,
                         seller_types=SELLER_TYPES,
                         transmission_types=TRANSMISSION_TYPES,
                         owner_types=OWNER_TYPES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = request.get_json()
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            data['name'],
            int(data['year']),
            int(data['km_driven']),
            data['fuel'],
            data['seller_type'],
            data['transmission'],
            data['owner'],
            float(data['mileage']),
            int(data['engine']),
            int(data['max_power']),
            int(data['seats'])
        ]], columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 
                     'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
        
        # Encode categorical variables
        input_data['owner'] = input_data['owner'].map(OWNER_MAPPING)
        input_data['fuel'] = input_data['fuel'].map(FUEL_MAPPING)
        input_data['seller_type'] = input_data['seller_type'].map(SELLER_MAPPING)
        input_data['transmission'] = input_data['transmission'].map(TRANSMISSION_MAPPING)
        input_data['name'] = input_data['name'].map(BRAND_MAPPING)
        
        # Check if model exists
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded properly'
            }), 500
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'success': True,
            'price': float(prediction[0]),
            'formatted_price': f"₹{prediction[0]:,.2f}"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚗 Car Price Predictor Server")
    print("="*50)
    print(f"✓ Server running at: http://localhost:5000")
    print(f"✓ Press CTRL+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)