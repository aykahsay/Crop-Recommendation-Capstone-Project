
# **🌾 Crop Recommendation System — Machine Learning Capstone Project**

**DSA3020 VA — United States International University–Africa**

<img width="915" alt="banner" src="https://github.com/user-attachments/assets/d6346dcd-5315-41e0-b7d8-06c474b3463e" />

---

## 📰 **Project Overview**

This project builds a full **end-to-end machine learning system** that recommends the most suitable crop to grow based on soil nutrients and environmental conditions.

The workflow follows the **CRISP-DM methodology**, covering:

* Business Understanding
* Data Understanding
* Data Preparation
* Modeling
* Evaluation
* Deployment

A user-friendly **Streamlit application** enables real-time crop recommendation from environmental inputs.

<img width="1280" alt="app" src="https://github.com/user-attachments/assets/d87400b9-15a0-47a5-b534-6e56bb1c00d6" />

---

## 🎯 **Objectives**

* Build a predictive model for crop recommendation
* Apply CRISP-DM methodology
* Compare various ML algorithms
* Deploy a functional Streamlit web app
* Demonstrate collaboration, version control, and reproducibility

---

## 📂 **Project Structure**

```
Crop_Recommendation_Project/
│
├── app/
│   ├── app.py                      # Streamlit app
│   └── requirements.txt            # Dependencies
│
├── saved_models/
│   ├── crop_recommendation_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── Crop_recommendation.csv
│
├── notebooks/
│   ├── 01_data_preprocessing_and_exploration.ipynb
│   ├── 02_model_training_and_selection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
│
├── reports/
│   ├── figures/
│   └── final_report.docx
│
├── README.md
└── .gitignore
```

---

## 🧠 **Dataset Description**

The dataset contains agronomic variables commonly used for crop suitability analysis.

| Feature         | Description               |
| --------------- | ------------------------- |
| **N**           | Nitrogen content in soil  |
| **P**           | Phosphorus content        |
| **K**           | Potassium content         |
| **temperature** | Temperature (°C)          |
| **humidity**    | Relative humidity (%)     |
| **ph**          | Soil pH                   |
| **rainfall**    | Rainfall (mm)             |
| **label**       | Recommended crop (target) |

[Kaggle-Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset?utm_source=chatgpt.com)
---

## 🤖 **Modeling Approach**

The following ML algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* XGBoost Classifier
📊 Evaluation Visuals

Train vs Test Accuracy

Train vs Test MAE

Confusion Matrices

## 📊 **Model Evaluation Visuals**

The analysis includes:

* Train vs Test Accuracy Plot
* Train vs Test MAE Plot
* Confusion Matrix (Multiclass)
* Feature Importance Plot

Performance Metrics:

| Model | Test Accuracy | Train Accuracy | MAE (Test) | Macro F1-score |
|-------|---------------|----------------|------------|----------------|
| **RTRFC** | 1.00 | 1.00 | 0.01 | 0.99 |
| **XGBoost (XGBC)** | 0.99 | 1.00 | 0.00 | 0.99 |
| **Random Forest (RFC)** | 0.99 | 0.99 | 0.12 | 0.98 |
| **SVM** | 0.97 | 0.99 | 0.14 | 0.95 |
| **Logistic Regression (LR)** | 0.96 | 0.97 | 0.30 | 0.95 |


Note: Dataset was confirmed balanced, and train-test split was done before scaling and encoding to avoid leakage.

<img width="3000" height="1800" alt="image" src="https://github.com/user-attachments/assets/2258081a-bb3a-4e92-acff-fec6ecb5057f" />

**[For more Figures click this link]**(https://github.com/aykahsay/Crop-Recommendation-Capstone-Project/tree/main/reports/figures)
---

#### **Selected Model**

The Reduced & Tuned Random Forest Classifier (RTRFC) was chosen for deployment due to its:

High accuracy

Robust handling of feature interactions

Interpretability through feature importance
## 🚀 **Deployment Instructions (Streamlit)**

### **1️⃣ Install dependencies**

```bash
pip install -r app/requirements.txt
```

### **2️⃣ Run the application**

```bash
streamlit run app/app.py
```

### **3️⃣ Enter the following inputs**

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

The app instantly returns the recommended crop.

---

## 💾 **Saved Artifacts**

| File                            | Purpose                             |
| ------------------------------- | ----------------------------------- |
| `crop_recommendation_model.pkl` | Tuned Random Forest Model with selected Feaytures|
| `scaler.pkl`                    | StandardScaler for feature scaling  |
| `label_encoder.pkl`             | For decoding predicted crop classes |

These files allow direct deployment without retraining the model.
### **Model Deployment**
[Random Forest Model](http://localhost:8501/)
---
### **Presentation**
[ppt](https://github.com/aykahsay/Crop-Recommendation-Capstone-Project/blob/main/Smart%20Crop%20Recommendation%20System.pptx)
## 👥 **Group Members (DSA3020 VA)**

* **[Muhia, Wilson Junior Wambugu](https://github.com/wambugushub)**
  - [Data Preparaing and Preprocessing](https://github.com/aykahsay/Crop-Recommendation-Capstone-Project/blob/main/notebooks/01_data_preprocessing_and_exploration.ipynb)
* **[Kahsay, Ambachow Ykalom](https://github.com/aykahsay)**
  - [Modleing and Evalaution Lead](https://github.com/aykahsay/Crop-Recommendation-Capstone-Project/blob/main/notebooks/02_model_training._model_selection.ipynb)
* **[Muhumed, Zakariya Shafi](https://github.com/Zakishafi)**
  - [Deployment and Documentaion Lead](https://github.com/aykahsay/Crop-Recommendation-Capstone-Project/blob/main/app/app4.py)
---

## 🛠 **Technology Stack**

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Streamlit
* Joblib
* Jupyter Notebook
* Git & GitHub

---
## Project Managament
[TO DO LIST](https://to-do.office.com/tasks/AQMkADY1NDcyNmI4LTNmMWEtNGU5My05NmE5LTI3ODliMjRjMDMwNAAuAAADNsPU94SCGEqAxR1CThjnQQEApiBwbSONxk_0ev_qutn1rQACDjI0jQAAAA==)
## 📎 **License**

This project is developed for academic use under the **USIU-A DSA3020 VA** course.

---
