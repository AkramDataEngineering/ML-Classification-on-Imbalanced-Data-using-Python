# ML-Classification-on-Imbalanced-Data-using-Python

# 📊 Classification on Imbalanced Data

## 🔎 Project Overview
In **Machine Learning**, imbalanced data occurs when one class (majority) heavily outnumbers the other (minority).  
This often leads to biased models that predict the majority class more accurately, while failing to identify the minority class — which is usually of greater interest (e.g., fraud detection, churn, claims).

This project demonstrates **handling imbalanced datasets** and building a robust classification model using **Python (pandas, scikit-learn, seaborn, matplotlib)**.

---

## ⚙️ Process Followed
1. **Data Exploration**
   - Load and inspect the insurance claims dataset (`58,592 rows × 41 features`).
   - Check for missing values, feature types, and target distribution.

2. **Exploratory Data Analysis (EDA)**
   - Visualized class imbalance (`claim_status` variable).
   - Analyzed distributions of numerical features like `subscription_length`, `vehicle_age`, `customer_age`.
   - Explored categorical variables such as `region_code`, `segment`, `fuel_type`.

3. **Handling Imbalance**
   - Applied **oversampling (resample)** on the minority class.
   - Balanced dataset: `54,844` samples in each class.

4. **Feature Selection**
   - Used **Random Forest Feature Importance** to identify key predictors:
     - `subscription_length`, `customer_age`, `vehicle_age`, `region_density`, etc.

5. **Model Training**
   - Trained **Random Forest Classifier** on oversampled data.
   - Evaluation metrics used: **Precision, Recall, F1-score, Accuracy**.

6. **Model Performance**
   - Achieved **99% accuracy** with balanced precision & recall.
   - High recall for minority class (claims), ensuring better detection.

---

## 📊 Results
- **Balanced model** capable of detecting claims effectively.  
- **Random Forest** proved robust for handling categorical + numerical features.  
- Model evaluation showed **F1-score = 0.99** for both classes.  

---

## 🛠 Tech Stack
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn  
- **Model:** Random Forest Classifier  
- **Resampling:** sklearn.utils.resample  

---

## ✅ Key Takeaways
- Imbalanced datasets require special treatment (oversampling, resampling, or algorithmic approaches).  
- Metrics like **Precision, Recall, F1, AUROC** are more reliable than Accuracy.  
- Random Forest + Oversampling achieved strong results for this dataset.  

---
