# Customer Churn Prediction Flask App

A web application for predicting customer churn using a trained Random Forest model with SMOTEENN.

## Features

- **Interactive Web Form**: User-friendly interface to input customer details
- **Real-time Prediction**: Instant churn prediction with confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **Model Integration**: Uses pre-trained Random Forest model with SMOTEENN

## Installation

### Local Development

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model File**: Make sure `model.sav` is in the project directory

### Render Deployment

1. **Connect to Render**: Link your GitHub repository to Render
2. **Configure Service**: 
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Python Version: 3.10.12
3. **Deploy**: Render will automatically deploy your app

## Usage

1. **Run the Flask Application**:
   ```bash
   python app.py
   ```

2. **Access the Web App**: Open your browser and go to `http://127.0.0.1:5000`

3. **Fill the Form**: Enter customer details in the form fields

4. **Get Prediction**: Click "Predict Churn" to see the result

## Model Details

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: SMOTEENN for handling class imbalance
- **Features**: 19 customer attributes including demographics, services, and billing information

## Form Fields

The application collects the following customer information:

### Demographics
- Gender
- Senior Citizen status
- Partner status
- Dependents

### Account Information
- Tenure (months)
- Contract type
- Paperless billing
- Payment method

### Services
- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming TV
- Streaming movies

### Billing
- Monthly charges
- Total charges

## Project Structure

```
customer_churn/
├── app.py                 # Flask application
├── model.sav             # Trained Random Forest model
├── tel_churn.csv         # Training dataset
├── requirements.txt      # Python dependencies
├── test_model.py         # Model testing script
├── README.md            # This file
├── templates/
│   └── index.html       # Web form template
└── static/
    └── style.css        # CSS styling
```

## Testing

Run the test script to verify model loading and preprocessing:

```bash
python test_model.py
```

## API Endpoints

- `GET /` - Homepage with prediction form
- `POST /predict` - Process form data and return prediction

## Prediction Output

The application returns:
- **Prediction**: "Will Churn" or "Will Not Churn"
- **Confidence**: Overall prediction confidence percentage
- **Churn Probability**: Specific probability of churn

## Troubleshooting

1. **Model Loading Issues**: Ensure `model.sav` exists and is properly formatted
2. **Dependencies**: Install all required packages from `requirements.txt`
3. **Port Issues**: If port 5000 is busy, modify the port in `app.py`

## Requirements

- Python 3.7+
- Flask 2.3.3
- pandas 2.0.3
- numpy 1.24.3
- scikit-learn 1.3.0
