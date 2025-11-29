# **Crop Recommendation System — Machine Learning Capstone Project**

**DSA3020 VA — United States International University–Africa**

---

## 📰 **Project Overview**

This project implements a full **end-to-end machine learning system** that recommends the most suitable crop to grow based on soil and environmental parameters. It follows the **CRISP-DM methodology**, covering business understanding, data preparation, modeling, evaluation, and deployment.

The system is deployed using **Streamlit** to allow users to input soil nutrient levels and climate conditions, and receive real-time crop recommendations.

---

## 🎯 **Objectives**

* Build a machine learning model that predicts the best crop to grow.
* Use CRISP-DM to guide project workflow.
* Compare multiple ML algorithms to identify the best performer.
* Deploy the final model through a simple, free, interactive web application.
* Demonstrate teamwork, version control skills, and reproducible research.

---

## 📂 **Project Structure**

```
Crop_Recommendation_Project/
│
├── app/
│   ├── app.py                      # Streamlit app for deployment
│   └── requirements.txt            # Libraries needed for deployment
│
├── saved_models/
│   ├── crop_recommendation_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── Crop_recommendation.csv     # Original dataset
│
├── notebooks/
│   ├── 01_data_epreprocessing_and_xploration.ipynb
│   ├── 02__model_training_and _selection.ipynb
│  
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
│
├── reports/
│   ├── figures/                    # Accuracy/MAE plots, confusion matrices
│   └── final_report.docx
│
├── README.md
└── .gitignore
```

---

## 🧠 **Dataset Description**

The dataset contains agricultural soil and environmental features:

| Feature         | Description                        |
| --------------- | ---------------------------------- |
| **N**           | Nitrogen content                   |
| **P**           | Phosphorus content                 |
| **K**           | Potassium content                  |
| **temperature** | Temperature in °C                  |
| **humidity**    | Relative humidity (%)              |
| **ph**          | Soil pH value                      |
| **rainfall**    | Rainfall in mm                     |
| **label**       | Recommended crop (target variable) |

---

## 🔧 **Modeling Approach**

The following models were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* XGBoost Classifier (XGBC)

### **Final Model Selected: XGBoost Classifier**

| Metric       | Test Score |
| ------------ | ---------- |
| **Accuracy** | **0.99**   |
| **MAE**      | **0.12**   |

XGBoost outperformed all other models and was chosen for deployment.

*Analogy about https://gemini.google.com/share/abef4af8514e*
---

## 📊 **Evaluation Visuals**

Plots include:

* Train vs Test Accuracy (side-by-side)
* Train vs Test MAE (side-by-side)
* Confusion Matrix
* Feature Importance

(All stored inside `reports/figures/`.)

---

## 🚀 **Deployment Instructions (Streamlit – Free)**

### **1️⃣ Install dependencies**

```
pip install -r app/requirements.txt
```

### **2️⃣ Run the Streamlit app**

```
streamlit run app/app.py
```

The app will open in your browser and allow users to input:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

And receive a crop recommendation instantly.

---

## 💾 **Saved Artifacts**

* `crop_recommendation_model.pkl` — trained XGBoost model
* `scaler.pkl` — StandardScaler used during preprocessing
* `label_encoder.pkl` — encoder for converting crop labels

These allow seamless inference during deployment.

---

## 👥 **Team Members**

List your group here:

Group Members

M## Group Members
- Muhia, Wilson Junior Wambugu
- [Muhumed, Zakariya Shafi](https://github.com/Zakishafi)
- [Kahsay, Ambachow Ykalom](https://github.com/aykahsay)


---

## 📘 **Technology Stack**

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

This project is for academic and educational purposes under the USIU-A DSA3020 VA course.

---
