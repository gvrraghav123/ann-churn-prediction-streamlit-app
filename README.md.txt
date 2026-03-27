# 🧠 Customer Churn Prediction using ANN + Streamlit

## 📌 Project Overview
This project predicts whether a bank customer is likely to churn (leave the bank) using an Artificial Neural Network (ANN) model trained on customer demographic and financial data.

The model is deployed using **Streamlit** for an interactive web interface.

---

## 🚀 Live Features
- User-friendly web UI (Streamlit)
- Real-time prediction
- ANN deep learning model (TensorFlow/Keras)
- Encoders for categorical variables
- Feature scaling for model consistency

---

## 🧾 Input Features

The model takes the following inputs:

| Feature | Description |
|--------|-------------|
| Credit Score | Customer credit score |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with bank |
| Balance | Account balance |
| Number of Products | Products owned |
| Has Credit Card | 0 = No, 1 = Yes |
| Active Member | 0 = No, 1 = Yes |
| Estimated Salary | Customer salary |
| Geography | France / Germany / Spain |

---

## 🧠 Model Architecture
- Type: Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Preprocessing:
  - Label Encoding (Gender)
  - One Hot Encoding (Geography)
  - Standard Scaling

---

## 📁 Project Structure
