
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

---

## 🤖 **Modeling Approach**

The following ML algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* **XGBoost Classifier (Final Model)**

---

## 🏆 **Final Selected Model: XGBoost Classifier**

| Metric       | Score    |
| ------------ | -------- |
| **Accuracy** | **0.99** |
| **MAE**      | **0.12** |

### ✔ Why XGBoost?

* Highest accuracy among all models
* Handles non-linear relationships
* Strong regularization (low overfitting)
* Clear feature importance for interpretation

---

## 📊 **Model Evaluation Visuals**

The analysis includes:

* Train vs Test Accuracy Plot
* Train vs Test MAE Plot
* Confusion Matrix (Multiclass)
* Feature Importance Plot

These plots are included in the `reports/figures/` directory.

---

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
| `crop_recommendation_model.pkl` | Trained XGBoost model               |
| `scaler.pkl`                    | StandardScaler for feature scaling  |
| `label_encoder.pkl`             | For decoding predicted crop classes |

These files allow direct deployment without retraining the model.
Model 
http://localhost:8513/
---

## 👥 **Group Members (DSA3020 VA)**

* **Muhia, Wilson Junior Wambugu**
* **[Muhumed, Zakariya Shafi](https://github.com/Zakishafi)**
* **[Kahsay, Ambachow Ykalom](https://github.com/aykahsay)**

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

## 📎 **License**

This project is developed for academic use under the **USIU-A DSA3020 VA** course.

---
