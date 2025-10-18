# 📊 End-to-End Customer Churn Prediction (ML + Flask + Render Deployment)

### 🧩 Overview

This project is a **complete end-to-end machine learning solution** that predicts customer churn for a **telecommunications company**.  
It covers the **entire data science lifecycle** — from **EDA (Exploratory Data Analysis)** and **data preprocessing** to **model building** and **deployment** as an **interactive Flask web application**.

The main goal is to analyze customer data, identify the **key drivers of churn**, and build a reliable model to **predict which customers are at risk of leaving**.

---

## 🖥️ Live Demo

An intuitive **Flask web application** allows users to input customer details (gender, tenure, monthly charges, contract type, etc.) and get:
- A **prediction** (Churn / Not Churn)
- A **confidence score** (churn probability)

📸 *Add a screenshot of your web app UI here:*  
![Web App UI](Screenshot%202025-10-19%20005934.jpg)

---

## 🛠️ Tech Stack

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

### 1️⃣ Data Exploration & EDA
**File:** `Churn_Analysis_EDA.ipynb`

- Performed exploratory data analysis to understand data distributions, correlations, and churn patterns.  
- Visualized key churn indicators such as **contract type**, **tech support availability**, and **monthly charges**.

### 2️⃣ Data Preprocessing & Feature Engineering
- Converted categorical features into numerical using **One-Hot Encoding**.  
- Binned the **tenure** feature for better segmentation.  
- Converted the target variable `Churn` → 1 (Yes) / 0 (No).  
- Handled missing values and data type conversions.

### 3️⃣ Model Building
**File:** `Churn_Analysis_Model_Building.ipynb`

- Tested multiple models: **Decision Tree**, **Random Forest**, etc.  
- Handled imbalance using **SMOTEENN**, improving recall on the minority (churn) class.  
- Final model: **Random Forest Classifier** with ~92% accuracy.  
- Saved the trained model as `model.sav` using **pickle**.

### 4️⃣ Web Application
**File:** `app.py`

- Built with **Flask** to serve the model.  
- Simple web form for user input and real-time prediction.  
- Integrated HTML templates and styled using basic CSS.

### 5️⃣ Deployment (Render)
- Configured for **Render** cloud deployment.  
- Added `render.yaml` and `requirements.txt` for automated build setup.  
- Live prediction API available once deployed.

---

## 📊 Key Insights from EDA

### 1️⃣ The Churn Rate is Imbalanced
A large portion of customers remain loyal, making churn prediction an **imbalanced classification problem**.  
![Churn Imbalance](Screenshot%202025-10-19%20013251.png)

---

### 2️⃣ Internet Service Matters
Customers with **Fiber Optic** internet have a higher churn rate — possibly due to pricing or performance issues.  
![Internet Service vs Churn](Screenshot%202025-10-19%20013344.png)

---

### 3️⃣ High Monthly Charges = High Churn
Customers paying more ($70–$100) are more likely to churn.  
![Monthly Charges vs Churn](Screenshot%202025-10-19%20013409.png)

---

### 4️⃣ Tech Support Retains Customers
Lack of tech support strongly correlates with churn — service quality is key to retention.  
![Tech Support vs Churn](Screenshot%202025-10-19%20013354.png)

---

### 5️⃣ Demographics Have Less Impact
Gender, partner status, and dependents show little effect on churn.  
![Gender vs Churn](Screenshot%202025-10-19%20013315.png)

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

## 🌐 Flask Application Structure

