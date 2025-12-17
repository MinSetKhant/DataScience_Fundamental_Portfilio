# DataScience_DataMining_Portfilio
DataScience_DataMining_Portfilio

# Healthcare Equity Explorer: 30-Day Readmission Prediction


**INFO 523 – Final Project**
**Author:** Min Set Khant
**Institution:** University of Arizona


---


## 1. Project Overview


This project develops a predictive model to identify patients at elevated risk of hospital readmission within 30 days using a Synthea-derived healthcare dataset. Beyond predictive accuracy, the project investigates potential equity implications across demographic and socioeconomic groups.


**Primary Evaluation Metric:** ROC AUC
**Modeling Objective:** Accurate and equitable prediction of 30-day readmission
**Final Model:** XGBoost with hyperparameter tuning and cross-validation


---


## 2. Repository Structure


A clean and fully reproducible organization is maintained throughout:


```
project-root/
│
├── data/
│   ├── train.csv
│   ├── dev.csv
│   ├── cleaned_train.csv
│   └── (test.csv will be added during test phase)
│
├── notebooks/
│   └── final_project_notebook.ipynb
│
├── src/
│   └── ds.py
│
├── models/
│   └── (model.pkl generated after running training)
│
└── README.md
```


---


## 3. Data Preprocessing


All preprocessing operations were conducted within the notebook to ensure transparency:


### Missing Data Handling


* Numerical features imputed using **median**.
* Categorical features imputed using **mode**.
* Dataset validated for missingness after imputation.


### Additional Cleaning


* Removal of duplicates
* Data type inspection and harmonization
* Categorization of age into four clinically interpretable groups:


 * Young (0–30)
 * Middle-aged (31–50)
 * Senior (51–70)
 * Elderly (71–90)


A cleaned dataset is stored as **`cleaned_train.csv`** for reproducibility.


---


## 4. Exploratory Data Analysis (EDA)


The exploratory analysis investigated demographic patterns, cost structures, chronic disease burden, and key correlates of readmission.


### Key Findings


* **Readmission Rate:** ≈63%, indicating moderate class imbalance.
* **Age:** Positive correlation with readmission (r ≈ 0.236).
* **Chronic conditions:** Multiple comorbidities are strongly associated with readmission.
* **Costs:** Encounter cost exhibits a negative correlation with readmission (r ≈ –0.156).
* **Medication burden:** Higher numbers of medications are associated with higher readmission risk.
* **Demographic variation:** Disparities exist across race, gender, and payer type.


These findings informed the feature engineering and modeling strategy.


---


## 5. Feature Engineering


Several clinically meaningful and interpretable features were constructed:


### Engineered Features


* **chronic_count:** Count of chronic conditions (hypertension, pain, diabetes, asthma, depression).
* **cost_ratio:** Total medication cost relative to encounter cost.
* **proc_ratio:** Ratio of procedures to medications.
* **high_risk_flag:** Indicator for age > 70 or ≥ 3 chronic conditions.
* **One-hot encoding** for gender, race, ethnicity, payer type, age group.


These features improved model flexibility and captured patient complexity.


---


## 6. Modeling and Evaluation


Two models were developed:


### 6.1 Baseline Model — Logistic Regression


* Standardized numeric features
* Class-balanced weighting
* ROC AUC: ~0.66
* Strength: interpretability
* Limitation: inability to capture nonlinear interactions


### 6.2 Final Model — XGBoost


Hyperparameters tuned via `GridSearchCV` (3-fold cross-validation):


* `max_depth`: 4–6
* `learning_rate`: 0.05–0.10
* `n_estimators`: 200–300
* `subsample`: 0.8–1.0
* `colsample_bytree`: 0.8–1.0


**Best Cross-Validation ROC AUC:** ~0.82–0.85
**Development Set ROC AUC:** similar performance
**F1 Score:** ~0.84


XGBoost was selected as the final model due to its strong discriminative performance and robustness.


---


## 7. Ethical Considerations


This project acknowledges the potential risk of perpetuating structural biases embedded in healthcare data. The inclusion of demographic and socioeconomic attributes is intended for identifying inequities, not reinforcing them.


Key principles applied:


* These features must not be used for punitive decision-making.
* Predictive outputs should complement, not replace, clinical judgment.
* Group-level performance disparities should be audited prior to deployment.
* Fairness adjustments should be considered if this model were operationalized.


The model is positioned as a decision-support tool, particularly useful for targeting care coordination resources to underserved populations.


---


## 8. Reproducibility


### Software Requirements


* Python 3.10+
* XGBoost
* scikit-learn
* pandas, numpy, seaborn, matplotlib
* joblib


Install all requirements using:


```
pip install -r requirements.txt
```


---


## 9. Running the Model (ds.py)


### Training


```
python src/ds.py train \
   --train_path data/train.csv \
   --dev_path data/dev.csv \
   --model_path models/model.pkl
```


### Prediction (only available during test phase)


```
python src/ds.py predict \
   --model_path models/model.pkl \
   --input_path data/test.csv \
   --output_path submission.csv
```


This ensures consistent, automated model training and inference.


---


## 10. Summary of Findings


The strongest predictors of readmission included:


* Advanced patient age
* High medication burden
* Multiple chronic conditions
* Treatment cost patterns
* Certain demographic indicators that may reflect healthcare inequity


The final tuned XGBoost model performed substantially better than the baseline, achieving strong ROC AUC and recall — supporting its use as a proactive risk stratification tool.


---





