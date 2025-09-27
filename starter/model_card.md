# 📊 Model Card – Census Income Classification

## 1. Model Details
- **Model name:** Census Income Classifier  
- **Model version:** 1.0  
- **Owner:** Husam Issa  
- **Algorithm:** Logistic Regression (from scikit-learn)  
- **Framework:** scikit-learn 1.1.1  
- **Date trained:** September 2025  
- **Location of model:** `starter/model/model.pkl`

This model predicts whether a person earns more than \$50,000/year based on demographic and employment features from the U.S. Census dataset.

---

## 2. Intended Use
The model is intended for **educational and demonstration purposes** in a machine learning deployment project.  
It shows how to build, evaluate, and deploy a classification model as an API.

- **Primary users:** Students, data scientists, ML engineers  
- **Use cases:**  
  - Example of model training and deployment  
  - Demonstrating ML inference via REST API  

---

## 3. Training Data
- **Source:** U.S. Census Bureau "Adult" dataset  
- **Size:** ~32,561 rows  
- **Split:** 80% training / 20% test  
- **Features:** Age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country  
- **Target:** Binary label – `>50K` or `<=50K`

---

## 4. Evaluation Data
The test set (20% of the total data) was used for evaluation.  
All preprocessing steps (encoding, label binarization) were applied identically to training and test data.

---

## 5. Metrics
The model was evaluated using standard classification metrics:

| Metric     | Value   |
|------------|---------|
| Precision  | 0.78    |
| Recall     | 0.74    |
| F1-score   | 0.76    |

*(Replace with your actual metrics from `train_model.py` output.)*

---

## 6. Ethical Considerations
- **Bias:** The model is trained on historical census data which may contain societal biases.  
- **Fairness:** Results might vary across demographic groups; use with caution in sensitive decision-making.  
- **Usage:** Not suitable for real-world decision-making without bias analysis and mitigation steps.

---

## 7. Caveats and Recommendations
- The model performance is dependent on the quality of the input features.  
- Retraining is recommended if the data distribution changes over time.  
- Additional fairness checks and feature importance analysis are recommended before production use.

---

✍️ *Prepared by Husam Issa – September 2025*
