import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# same data preparation as train_model.py
company = pd.read_csv("Company data.csv")
company.columns = (
    company.columns.str.replace(" ", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "perc", regex=False)
    .str.replace("/", "_to_", regex=False)
)
y = np.where(company["Networth_Next_Year"] <= 0, 1, 0)
X = company.drop(columns=["Networth_Next_Year"])
X = X.drop(columns=[
    "Deposits_accepted_by_commercial_banks", "PE_on_BSE", "Investments",
    "Other_income", "Contingent_liabilities", "Deferred_tax_liability",
    "Income_from_financial_services", "Num",
])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2)

models = {
    "1. RF + SMOTE (current)": Pipeline([
        ("imp", KNNImputer(n_neighbors=11)),
        ("smote", SMOTE(random_state=33)),
        ("m", RandomForestClassifier(n_estimators=300, random_state=2, n_jobs=-1)),
    ]),
    "2. RF + class_weight": Pipeline([
        ("imp", KNNImputer(n_neighbors=11)),
        ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     min_samples_leaf=3, random_state=2, n_jobs=-1)),
    ]),
    "3. Gradient boosting + class_weight": HistGradientBoostingClassifier(
        class_weight="balanced", max_depth=4, learning_rate=0.05, random_state=2),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

for name, m in models.items():
    p = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    b = np.argmax(f1[:-1])
    print(name)
    print("  ROC-AUC:", round(roc_auc_score(y_tr, p), 3),
          "| PR-AUC:", round(average_precision_score(y_tr, p), 3))
    print("  Best threshold:", round(thr[b], 2),
          "| precision:", round(prec[b], 2), "| recall:", round(rec[b], 2))