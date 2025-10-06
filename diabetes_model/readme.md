# 💉 Diabetes Prediction using Machine Learning

An end-to-end Machine Learning project that predicts **Diabetes occurrence (Yes/No)** using the **PIMA Indians Diabetes Dataset**.  
This project is developed as part of my **CodeAlpha Internship**, applying data science techniques to solve a real-world healthcare challenge.

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

Diabetes is one of the most common chronic diseases worldwide. **Early detection** and **accurate prediction** can play a crucial role in preventing severe complications.  
This project uses **machine learning models** to predict whether a person is likely to have diabetes based on diagnostic and physiological parameters.

The pipeline includes:
- Data Cleaning & Missing Value Imputation  
- Outlier Detection and Handling  
- Exploratory Data Analysis (EDA)  
- Model Training, Hyperparameter Tuning, and Evaluation  
- Cross-Validation & Feature Importance Analysis  
- Model Saving and Dummy Predictions  

---

## 📊 Dataset Information

- **Dataset:** PIMA Indians Diabetes Dataset  
- **Source:** [Kaggle - Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
- **Samples:** 768  
- **Features:** 8 independent variables + 1 target variable  
- **Target Variable:**  
  - `0` = No Diabetes  
  - `1` = Diabetes  

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age of the person |

---

## 🚀 Project Workflow

1. **Data Preprocessing**
   - Replaced missing values using **KNN Imputer**  
   - Detected and treated **outliers**  
   - Scaled features using **StandardScaler**

2. **Exploratory Data Analysis (EDA)**
   - Checked data distribution using **histograms and KDE plots**  
   - Visualized **correlation heatmap**  
   - Analyzed class balance and feature relationships  

3. **Model Building**
   - Logistic Regression  
   - Random Forest  
   - XGBoost (Gradient Boosted Trees)  
   - Support Vector Machine (SVM)

4. **Model Evaluation**
   - Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**  
   - Used confusion matrix and ROC curves for visualization  

5. **Model Optimization**
   - Hyperparameter tuning using **GridSearchCV**  
   - Performed **Cross-Validation (CV)** for robustness  

6. **Feature Importance Analysis**
   - Identified most impactful features like **Glucose**, **BMI**, and **Age**  

7. **Model Saving & Prediction**
   - Saved best model and scaler using `joblib`  
   - Tested with dummy patient data for prediction verification  

---

## 🛠 Tech Stack

- **Languages:** Python  
- **Libraries:**  
  - pandas, numpy, matplotlib, seaborn  
  - scikit-learn  
  - xgboost  
  - joblib  
- **Tools:**  
  - Jupyter Notebook  
  - Git & GitHub  

---

## 🏆 Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression   | 0.79     | 0.76      | 0.72   | 0.74     | 0.82    |
| Random Forest         | 0.83     | 0.80      | 0.77   | 0.78     | 0.86    |
| SVM                   | 0.81     | 0.78      | 0.73   | 0.75     | 0.84    |
| XGBoost (Tuned)       | **0.85** | **0.83**  | **0.81** | **0.82** | **0.88** |

✅ **XGBoost achieved the best overall performance**, showing strong balance between recall and precision.

---

## 🔍 Key Insights

- **Glucose**, **BMI**, and **Age** are the most influential features.  
- Imputing missing values with **KNN Imputer** improved data quality.  
- **Hyperparameter tuning** significantly boosted model accuracy and AUC.  
- Model generalization validated through **cross-validation**.  

---

## 🌱 Future Work

- Incorporate additional clinical and lifestyle features.  
- Develop a **Streamlit web app** for real-time diabetes prediction.  
- Deploy on **cloud platforms (AWS / GCP)** for accessibility.  
- Add **Explainable AI (XAI)** to enhance interpretability for healthcare use.  

---

## 📚 References

- [Kaggle - PIMA Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
- [Pandas Documentation](https://pandas.pydata.org/)  
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)  

---

## 👩‍💻 Author

**Adina Abrar**  
📍 Computer Science Student | Data Science & AI Enthusiast  
✨ Completed under **CodeAlpha Internship Program**

🔗 [LinkedIn](https://www.linkedin.com/in/adina-abrar/) | [GitHub](https://github.com/Adina-Abrar)
