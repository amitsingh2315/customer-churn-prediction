#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
"""

import pickle
import pandas as pd
import numpy as np

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        with open('model.sav', 'rb') as file:
            model_data = pickle.load(file)
            print("✓ Model file loaded successfully")
            print(f"Model data type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print("✓ Model data is a dictionary")
                print(f"Available keys: {list(model_data.keys())}")
                
                # Try to find the model
                model = None
                for key in ['model', 'classifier', 'rf_model', 'random_forest']:
                    if key in model_data:
                        model = model_data[key]
                        print(f"✓ Found model with key: {key}")
                        print(f"Model type: {type(model)}")
                        break
                
                if model is None:
                    print("⚠ No model found in dictionary")
                    return False
                    
            else:
                print("✓ Model data is a single object")
                model = model_data
                print(f"Model type: {type(model)}")
            
            # Test basic model methods
            if hasattr(model, 'predict'):
                print("✓ Model has predict method")
            else:
                print("⚠ Model missing predict method")
                
            if hasattr(model, 'predict_proba'):
                print("✓ Model has predict_proba method")
            else:
                print("⚠ Model missing predict_proba method")
                
            return True
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_preprocessing():
    """Test the preprocessing function with sample data"""
    try:
        # Import the preprocessing function
        from app import preprocess_data
        
        # Create sample data
        sample_data = {
            'SeniorCitizen': [0],
            'tenure': [24],
            'MonthlyCharges': [70.0],
            'TotalCharges': [1680.0],
            'gender': ['Male'],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check']
        }
        
        df = pd.DataFrame(sample_data)
        processed = preprocess_data(df)
        
        print("✓ Preprocessing function works")
        print(f"Processed data shape: {processed.shape}")
        print(f"Expected shape: (1, 50)")  # Based on the CSV columns
        
        return True
        
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return False

if __name__ == "__main__":
    print("Testing Customer Churn Prediction Model")
    print("=" * 50)
    
    print("\n1. Testing model loading...")
    model_ok = test_model_loading()
    
    print("\n2. Testing preprocessing...")
    preprocess_ok = test_preprocessing()
    
    print("\n" + "=" * 50)
    if model_ok and preprocess_ok:
        print("✓ All tests passed! The Flask app should work correctly.")
    else:
        print("⚠ Some tests failed. Check the issues above.")
