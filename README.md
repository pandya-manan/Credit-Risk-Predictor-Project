# 💳 Credit Risk Prediction System

## 🚀 Project Overview

This project builds an end-to-end machine learning system to predict the probability of loan default based on customer financial and behavioral data.

The goal is to assist in identifying high-risk applicants and support better credit decision-making.

---

## 📊 Key Features

* Feature engineering (loan-to-income, credit utilization per income)
* Logistic Regression with Bayesian Optimization
* XGBoost model for high-performance prediction
* SHAP-based explainability (model interpretation)
* Interactive Streamlit web application
* End-to-end pipeline from data processing to deployment

---

## 🧠 Tech Stack

* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP
* **Frontend:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## ⚙️ How It Works

1. Data preprocessing and cleaning
2. Feature engineering to capture financial behavior
3. Model training using Logistic Regression and XGBoost
4. Hyperparameter tuning (Bayesian Optimization & GridSearchCV)
5. Model evaluation using ROC-AUC
6. Deployment with Streamlit for real-time predictions

---

## 📈 Model Performance

* Logistic Regression (tuned): ~0.94 ROC-AUC
* XGBoost (tuned): ~0.96 ROC-AUC

> Note: Performance is influenced by dataset characteristics and feature engineering.

---

## 🌐 Live Demo

👉 [Add your deployed Streamlit link here]

---

## 💻 GitHub Repository

👉 [Your repository link]

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

---

## 📁 Project Structure

```
credit-risk-ml/
│
├── app/
│   └── app.py
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   ├── numeric_cols.pkl
│   └── means.csv
│
├── notebooks/
│   └── credit_risk.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🧠 Key Learnings

* Importance of preventing data leakage in ML models
* Building consistent training and inference pipelines
* Feature engineering for financial risk modeling
* Deploying ML models into production using Streamlit

---

## 📌 Future Improvements

* Add SHAP explainability inside the web app
* Improve UI/UX for better user interaction
* Integrate API-based backend for scalability
* Add user authentication and reporting features

---

## 🤝 Acknowledgements

This project was built as part of hands-on learning in machine learning and deployment.

---

## 📬 Contact

pandyamanan100@gmail.com

---
