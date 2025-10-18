from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure for production
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Load the trained model
try:
    with open('model.sav', 'rb') as file:
        model_data = pickle.load(file)
        # Handle different model saving formats
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data.get('classifier', model_data.get('rf_model')))
            preprocessor = model_data.get('preprocessor', model_data.get('preprocessor_pipeline'))
        else:
            # If it's just the model object
            model = model_data
            preprocessor = None
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        if preprocessor:
            print(f"Preprocessor type: {type(preprocessor)}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

# Define the original categorical columns (before one-hot encoding)
categorical_columns = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Define numerical columns
numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# All original columns in order
all_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 
               'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form
        
        # Create a dictionary with the form data
        customer_data = {}
        
        # Handle numerical fields
        customer_data['SeniorCitizen'] = int(form_data.get('SeniorCitizen', 0))
        customer_data['tenure'] = int(form_data.get('tenure', 0))
        customer_data['MonthlyCharges'] = float(form_data.get('MonthlyCharges', 0))
        customer_data['TotalCharges'] = float(form_data.get('TotalCharges', 0))
        
        # Handle categorical fields
        customer_data['gender'] = form_data.get('gender', '')
        customer_data['Partner'] = form_data.get('Partner', '')
        customer_data['Dependents'] = form_data.get('Dependents', '')
        customer_data['PhoneService'] = form_data.get('PhoneService', '')
        customer_data['MultipleLines'] = form_data.get('MultipleLines', '')
        customer_data['InternetService'] = form_data.get('InternetService', '')
        customer_data['OnlineSecurity'] = form_data.get('OnlineSecurity', '')
        customer_data['OnlineBackup'] = form_data.get('OnlineBackup', '')
        customer_data['DeviceProtection'] = form_data.get('DeviceProtection', '')
        customer_data['TechSupport'] = form_data.get('TechSupport', '')
        customer_data['StreamingTV'] = form_data.get('StreamingTV', '')
        customer_data['StreamingMovies'] = form_data.get('StreamingMovies', '')
        customer_data['Contract'] = form_data.get('Contract', '')
        customer_data['PaperlessBilling'] = form_data.get('PaperlessBilling', '')
        customer_data['PaymentMethod'] = form_data.get('PaymentMethod', '')
        
        # Create DataFrame
        df = pd.DataFrame([customer_data])
        
        # Reorder columns to match training data
        df = df[all_columns]
        
        # Preprocess the data using the same preprocessor from training
        if preprocessor is not None:
            processed_data = preprocessor.transform(df)
        else:
            # Fallback preprocessing if preprocessor is not available
            processed_data = preprocess_data(df)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            # Get confidence score
            confidence = max(prediction_proba) * 100
            
            result = {
                'prediction': 'Will Churn' if prediction == 1 else 'Will Not Churn',
                'confidence': round(confidence, 2),
                'churn_probability': round(prediction_proba[1] * 100, 2)
            }
        else:
            result = {
                'prediction': 'Model not available',
                'confidence': 0,
                'churn_probability': 0
            }
        
        return render_template('index.html', result=result, form_data=customer_data)
        
    except Exception as e:
        error_message = f"Error processing prediction: {str(e)}"
        return render_template('index.html', error=error_message)

def preprocess_data(df):
    """Fallback preprocessing function to match the training data format"""
    processed_df = df.copy()
    
    # Handle missing values
    processed_df = processed_df.fillna(0)
    
    # Convert to numeric where needed
    for col in numerical_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Create tenure groups to match training data
    def get_tenure_group(tenure):
        if tenure <= 12:
            return '1 - 12'
        elif tenure <= 24:
            return '13 - 24'
        elif tenure <= 36:
            return '25 - 36'
        elif tenure <= 48:
            return '37 - 48'
        elif tenure <= 60:
            return '49 - 60'
        else:
            return '61 - 72'
    
    processed_df['tenure_group'] = processed_df['tenure'].apply(get_tenure_group)
    
    # One-hot encode categorical variables to match the training data format
    # Based on the CSV structure, we need to create the exact same columns
    processed_df = pd.get_dummies(processed_df, columns=categorical_columns + ['tenure_group'], drop_first=False)
    
    # Ensure we have all the expected columns from the training data
    expected_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'Churn',
        'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
        'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
        'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
    ]
    
    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    # Reorder columns to match training data
    processed_df = processed_df[expected_columns]
    
    # Remove the 'Churn' column as it's the target variable
    if 'Churn' in processed_df.columns:
        processed_df = processed_df.drop('Churn', axis=1)
    
    return processed_df.values

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='127.0.0.1', port=5000)
else:
    # For production deployment (Render)
    # The app will be served by gunicorn
    pass
