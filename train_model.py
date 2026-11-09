import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

THRESHOLD = 0.43   # chosen with cross-validation on the training data

# 1. Load data and clean column names
company = pd.read_csv("Company data.csv")
company.columns = (
    company.columns.str.replace(" ", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "perc", regex=False)
    .str.replace("/", "_to_", regex=False)
)

# 2. Target and features
y = np.where(company["Networth_Next_Year"] <= 0, 1, 0)
X = company.drop(columns=["Networth_Next_Year"])
X = X.drop(columns=[
    "Deposits_accepted_by_commercial_banks", "PE_on_BSE", "Investments",
    "Other_income", "Contingent_liabilities", "Deferred_tax_liability",
    "Income_from_financial_services", "Num",
])

# 3. Split, train
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=2
)

final_model = Pipeline([
    ("imputer", KNNImputer(n_neighbors=11)),
    ("rf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        min_samples_leaf=3, random_state=2, n_jobs=-1)),
])
final_model.fit(X_tr, y_tr)

# 4. Final check on the untouched test set (look at this once)
proba = final_model.predict_proba(X_te)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
print(classification_report(y_te, pred))
print("ROC-AUC:", round(roc_auc_score(y_te, proba), 3))
print("PR-AUC:", round(average_precision_score(y_te, proba), 3))

# 5. Save model, features and threshold together
joblib.dump(
    {"model": final_model, "features": X.columns.tolist(), "threshold": THRESHOLD},
    "default_model.pkl",
)
print("Saved default_model.pkl. Number of features:", X.shape[1])