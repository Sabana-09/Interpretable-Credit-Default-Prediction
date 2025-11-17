📘 Interpretable Credit Default Prediction using XGBoost & SHAP

This project builds a complete Credit Default Risk Prediction System using the UCI Credit Card Dataset, applying advanced preprocessing, XGBoost modeling, hyperparameter tuning, and SHAP explainability for global and local interpretation of model behavior.

The goal is not only to achieve strong predictive performance but also to provide transparent, interpretable insights that are valuable in real-world financial decision-making, such as loan approvals, risk scoring, and fairness evaluation.

🧠 Project Highlights
✔️ End-to-End ML Pipeline

Data loading

Preprocessing (missing values, encoding, scaling)

XGBoost classification

GridSearchCV tuning

Performance evaluation

✔️ Explainable AI

Traditional feature importance (XGBoost gain)

SHAP global importance

SHAP local explanations (force & waterfall plots)

Interpretation of decisions for individual applicants

✔️ Fully Reproducible

All outputs are automatically saved to /outputs/ so the analysis can be reproduced or audited easily.

✔️ Ready for GitIngest

This repository follows the exact structure required for automated submission.

📂 Repository Structure
credit_shap_project/
│
├── README.md                     # Project documentation (this file)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Clean ignore rules
│
├── notebook/
│   └── credit_default_shap.ipynb # Full Colab notebook with complete pipeline
│
├── data/
│   └── UCI_Credit_Card.csv       # Dataset (from Kaggle / UCI)
│
├── outputs/
│   ├── model.joblib              # Saved XGBoost model
│   ├── metrics.json              # Accuracy, F1, ROC-AUC, precision, recall
│   ├── predictions.csv           # True vs predicted labels
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_local_1.png
│   ├── shap_local_2.png
│   ├── shap_local_3.png
│   ├── shap_force_1.html
│   ├── shap_force_2.html
│   ├── shap_force_3.html
│   ├── technical_summary.txt
│   ├── local_shap_interpretation.txt
│   └── final_business_analysis.txt
│
└── src/                          # optional scripts
    ├── train_model.py
    ├── preprocess.py
    └── shap_analysis.py

📊 Dataset Used

Dataset: UCI Credit Card Clients Dataset
Source: UCI Machine Learning Repository / Kaggle mirror
Task: Predict next-month default (binary classification)
Rows: 30,000
Features: Demographics, bill amounts, repayment history, credit limits, etc.

This dataset is ideal for:

Realistic credit risk modeling

XGBoost on mixed categorical + numerical data

SHAP-based explainability

Fairness and bias analysis

⚙️ How to Run the Project (Google Colab)

Upload the repository or open the notebook:

notebook/credit_default_shap.ipynb


Mount Google Drive

Place dataset into:

/content/drive/MyDrive/credit_shap_project/data/


Run all cells in order

All outputs—including SHAP plots—will be saved automatically to:

/outputs/

📈 Model Performance Metrics (saved in metrics.json)

Includes:

Accuracy

F1 score

Precision & Recall

ROC-AUC

Confusion Matrix

These provide a complete evaluation of model predictive power.

🔍 Explainability Using SHAP

The project includes:

✔ Global SHAP Summary Plot

Shows how features contribute to the overall model:

Top risk-inducing variables

Protective variables

Interaction behaviors

✔ Local SHAP Plots for Individual Customers

Waterfall and Force plots for 3 cases:

True Positive

False Positive

False Negative
(or the closest available samples)

These help understand why the model decided someone might default.

📝 Included Reports

Inside /outputs/ you will find:

✔ technical_summary.txt

Summarizes:

preprocessing

model tuning

evaluation

feature importance vs SHAP

✔ local_shap_interpretation.txt

Explains 3 local SHAP plots:

Top contributing risk factors

Top mitigating factors

✔ final_business_analysis.txt

≤500-word high-level analysis covering:

Risk drivers

Fairness considerations

Business implications

Recommendations for lenders

🚀 Technologies Used

Python 3

Pandas

NumPy

Scikit-learn

XGBoost

SHAP

Matplotlib & Seaborn

Google Colab

Joblib