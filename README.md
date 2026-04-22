Predictive Modeling & Risk Scoring for Bank Customer Churn

Project Overview
This project develops a machine learning-based predictive model to estimate customer churn probability in a European banking dataset. The goal is to identify customers at risk of leaving the bank and generate actionable insights for retention strategies.

By applying classification algorithms and model evaluation techniques, the system helps banks shift from reactive churn analysis to proactive customer retention.

 Objectives
- Predict customer churn using machine learning models  
- Generate churn probability scores for risk segmentation  
- Identify key drivers influencing customer churn  
- Improve interpretability using feature importance analysis  
- Support data-driven retention strategies for banks  

Dataset Description
The dataset contains customer information such as:

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Active Membership Status  
- Estimated Salary  
- Target Variable: **Exited (Churn: 1, Not Churn: 0)**  

---

 Methodology

**1. Data Preprocessing**
- Removed irrelevant columns (CustomerId, Surname)
- Handled categorical variables using one-hot encoding
- Feature scaling using StandardScaler

### 2. Model Building
- Logistic Regression (baseline)
- Decision Tree Classifier
- Random Forest Classifier (final model)

### 3. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  

---

##  Results
- Random Forest performed best among all models  
- Achieved strong ROC-AUC performance (~0.85)  
- Identified key churn drivers such as:
  - Age
  - Account Balance
  - Number of Products
  - Activity Status  

---

##  Visualizations
- Confusion Matrix  
- ROC Curve  
- Feature Importance Graph  

---

##  Key Insights
- Older customers show higher churn tendency  
- Inactive customers are more likely to leave  
- Product usage significantly impacts retention  
- Account balance is a strong predictor of churn  

---

## 🛠️ Tools & Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook (Google Colab)  

---

##  Future Improvements
- Hyperparameter tuning for better performance  
- Use of XGBoost / LightGBM models  
- Deployment using Streamlit dashboard  
- SHAP analysis for deeper explainability  

---

## 📌 Conclusion
This project demonstrates how machine learning can be used to predict customer churn and support proactive decision-making in banking systems. It helps improve customer retention strategies through data-driven insights.
