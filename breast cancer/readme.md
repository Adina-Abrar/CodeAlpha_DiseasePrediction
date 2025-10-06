# 🩺 Breast Cancer Prediction using Machine Learning

An end-to-end Machine Learning project that predicts **Breast Cancer diagnosis (Benign vs Malignant)** using the **Breast Cancer Wisconsin Dataset**.  
This project is developed as part of my **CodeAlpha Internship** to apply data science techniques to a real-world healthcare problem.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Project Workflow](#-project-workflow)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)

---

## 🧠 Project Overview

Breast cancer is one of the most common cancers in women, and **early detection can save lives**.  
This project uses **machine learning algorithms** to classify tumors as **Benign (0)** or **Malignant (1)** based on diagnostic features from cell nuclei.

The pipeline includes:
- Data Preprocessing & Feature Scaling  
- Exploratory Data Analysis (EDA)  
- Baseline Model Training  
- Model Evaluation & Comparison  
- Robustness Check with Cross-Validation  
- Feature Importance Analysis  
- Model Saving & Dummy Predictions

---

## 📊 Dataset Information

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Samples:** 569  
- **Features:** 30 numerical features (e.g., radius, texture, smoothness, etc.)  
- **Target:**  
  - `0` = Benign  
  - `1` = Malignant

---

## 🚀 Project Workflow

1. **Data Preprocessing**
   - Encoding categorical variables  
   - Feature scaling using `StandardScaler`  
   - Handling missing values (if any)  

2. **Exploratory Data Analysis**
   - Distribution plots  
   - Correlation heatmaps  
   - Outlier detection using boxplots  

3. **Model Building**
   - Logistic Regression  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - XGBoost  

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC  
   - Comparative bar charts for metrics  
   - ROC curves for performance visualization  

5. **Cross-Validation**
   - 5-fold CV to evaluate model robustness  

6. **Feature Importance Analysis**
   - Identifying key predictors affecting the classification outcome  

7. **Model Saving & Deployment Prep**
   - Best model & scaler saved using `joblib`  
   - Predictions made on dummy patient data

---

## 🛠 Tech Stack

- **Languages:** Python  
- **Libraries:**  
  - pandas, numpy, matplotlib, seaborn  
  - scikit-learn  
  - xgboost  
  - joblib (for model persistence)  
- **Tools:**  
  - Jupyter Notebook  
  - Git & GitHub

---

## 🏆 Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression   | 0.956    | High      | High   | High     | 0.96    |
| Random Forest         | 0.963    | High      | High   | High     | 0.97    |
| SVM                   | 0.917    | Moderate  | Moderate | Moderate | 0.91  |
| XGBoost               | **0.967**| High      | High   | High     | **0.98** |

✅ **XGBoost achieved the highest accuracy and ROC-AUC**, making it the best model for this dataset.

---

## 🔍 Key Insights

- Features like **`radius_mean`**, **`texture_mean`**, and **`perimeter_mean`** are strong predictors of cancer type.  
- **Ensemble models (XGBoost, Random Forest)** perform slightly better than linear models.  
- Cross-validation confirmed the model’s **stability and reliability**.

---

## 🌱 Future Work

- Integrate **SHAP/LIME** for explainable AI to improve clinical trust.  
- Train on larger or more diverse datasets for generalization.  
- Build a **user-friendly web app** (e.g., using Streamlit or Flask) for real-time predictions.  
- Deploy the model on cloud platforms (e.g., Google Cloud / AWS).

---

## 📚 References

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)

---

## 👩‍💻 Author

**Adina Abrar**  
📍 Computer Science Student | Data Science & AI Enthusiast  
✨ Completed under **CodeAlpha Internship Program**  

🔗 [LinkedIn](https://www.linkedin.com/in/adina-abrar/) | [GitHub](https://github.com/Adina-Abrar)
