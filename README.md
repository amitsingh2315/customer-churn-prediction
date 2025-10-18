#  End-to-End Customer Churn Prediction (ML + Flask + Render Deployment)

<img width="1024" height="366" alt="image" src="https://github.com/user-attachments/assets/cd9b0e14-705b-4743-9c26-adb67ebbfbc2" />


###  Overview

This project is a **complete end-to-end machine learning solution** that predicts customer churn for a **telecommunications company**.  
It covers the **entire data science lifecycle** — from **EDA (Exploratory Data Analysis)** and **data preprocessing** to **model building** and **deployment** as an **interactive Flask web application**.

The main goal is to analyze customer data, identify the **key drivers of churn**, and build a reliable model to **predict which customers are at risk of leaving**.

---

##  Live Demo

An intuitive **Flask web application** allows users to input customer details (gender, tenure, monthly charges, contract type, etc.) and get:
- A **prediction** (Churn / Not Churn)
- A **confidence score** (churn probability)

<img width="1631" height="1558" alt="Screenshot 2025-10-19 005934" src="https://github.com/user-attachments/assets/71169583-b773-457d-b98d-033b12d2ad82" />


---

##  Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| **Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Imbalanced Data Handling** | Imbalanced-learn (SMOTEENN) |
| **Web Framework** | Flask |
| **Model Serialization** | Pickle |
| **Deployment** | Render |

---

## 📂 Project Workflow

### 1. Data Exploration & EDA
**File:** `Churn_Analysis_EDA.ipynb`

- Performed exploratory data analysis to understand data distributions, correlations, and churn patterns.  
- Visualized key churn indicators such as **contract type**, **tech support availability**, and **monthly charges**.

### 2️ Data Preprocessing & Feature Engineering
- Converted categorical features into numerical using **One-Hot Encoding**.  
- Binned the **tenure** feature for better segmentation.  
- Converted the target variable `Churn` → 1 (Yes) / 0 (No).  
- Handled missing values and data type conversions.

### 3️ Model Building
**File:** `Churn_Analysis_Model_Building.ipynb`

- Tested multiple models: **Decision Tree**, **Random Forest**, etc.  
- Handled imbalance using **SMOTEENN**, improving recall on the minority (churn) class.  
- Final model: **Random Forest Classifier** with ~92% accuracy.  
- Saved the trained model as `model.sav` using **pickle**.

### 4️ Web Application
**File:** `app.py`

- Built with **Flask** to serve the model.  
- Simple web form for user input and real-time prediction.  
- Integrated HTML templates and styled using basic CSS.

### 5️ Deployment (Render)
- Configured for **Render** cloud deployment.  
- Added `render.yaml` and `requirements.txt` for automated build setup.  
- Live prediction API available once deployed.

---

##  Key Insights from EDA

### 1️1 The Churn Rate is Imbalanced
A large portion of customers remain loyal, making churn prediction an **imbalanced classification problem**.  
<img width="996" height="659" alt="image" src="https://github.com/user-attachments/assets/0727fec6-db59-494d-a13a-807b4c4a9736" />


---

### 2️ Internet Service Matters
Customers with **Fiber Optic** internet have a higher churn rate — possibly due to pricing or performance issues.  
<img width="1669" height="877" alt="Screenshot 2025-10-19 013334" src="https://github.com/user-attachments/assets/6ac48837-7e0e-42b4-b3b3-36f3c650be21" />


---

### 3️⃣ High Monthly Charges = High Churn
Customers paying more ($70–$100) are more likely to churn.  
<img width="1669" height="877" alt="Screenshot 2025-10-19 013334" src="https://github.com/user-attachments/assets/1ae15b30-92b9-4fea-8b05-46f151d6ce64" />


---

### 4️⃣ Tech Support Retains Customers
Lack of tech support strongly correlates with churn — service quality is key to retention.  
<img width="1669" height="877" alt="Screenshot 2025-10-19 013334" src="https://github.com/user-attachments/assets/ed86fe60-f5c6-4a00-b308-3b57099a9a99" />


---

### 5️⃣ Demographics Have Less Impact
Gender, partner status, and dependents show little effect on churn.  

---

## ⚙️ Model Summary

| Step | Description |
|------|--------------|
| **Algorithm** | Random Forest Classifier |
| **Imbalance Handling** | SMOTEENN |
| **Accuracy** | ~92% |
| **Precision / Recall** | High for churned customers |
| **Output File** | `model.sav` |

---

## Output:

✅ Prediction: Will Not Churn
🔢 Churn Probability: 14.23%

