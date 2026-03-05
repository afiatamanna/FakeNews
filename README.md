# Fake News Detection (Machine Learning Project)

A machine learning project that classifies news articles as **Real** or **Fake** based on their text content.

## 📌 Overview
This project trains a text classification model using **TF-IDF vectorization** and **Logistic Regression**.  
It reads a dataset from a CSV file, trains the model, evaluates performance, and saves both the metrics and trained model.

Optionally, it can generate **SHAP explanations** to show which words influence the model's predictions.

---

## ⚙️ Features
- Text preprocessing using **TF-IDF**
- Classification using **Logistic Regression**
- Evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Confusion matrix counts
- Model persistence with **joblib**
- Optional **SHAP explanations** for model interpretability

---

## 🛠️ Tech Stack
- **Python**
- **pandas**
- **scikit-learn**
- **joblib**
