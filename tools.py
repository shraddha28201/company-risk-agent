import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- load model and data ----------
bundle = joblib.load("default_model.pkl")
MODEL, FEATURES, THRESHOLD = bundle["model"], bundle["features"], bundle["threshold"]

df = pd.read_csv("Company data.csv")
df.columns = (
    df.columns.str.replace(" ", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "perc", regex=False)
    .str.replace("/", "_to_", regex=False)
)
df["Num"] = df["Num"].astype(int)

# the 6 features the model finds most important
importances = MODEL.named_steps["rf"].feature_importances_
TOP = [FEATURES[i] for i in importances.argsort()[::-1][:6]]

# ---------- policy search ----------
rules = [line.strip() for line in open("policy.txt", encoding="utf-8") if line.strip()]
vectorizer = TfidfVectorizer(stop_words="english")
rule_vectors = vectorizer.fit_transform(rules)


def _find(company_num):
    return df[df["Num"] == int(company_num)]


def _clean(value):
    return None if pd.isna(value) else round(float(value), 2)


# ---------- Tool 1: company lookup ----------
def get_company(company_num):
    row = _find(company_num)
    if row.empty:
        return {"error": f"Company {company_num} not found."}
    return {"company_num": int(company_num),
            "key_financials": {col: _clean(row.iloc[0][col]) for col in TOP}}


# ---------- Tool 2: risk score ----------
def risk_score(company_num):
    row = _find(company_num)
    if row.empty:
        return {"error": f"Company {company_num} not found."}
    score = float(MODEL.predict_proba(row[FEATURES])[0, 1])
    if score >= THRESHOLD:
        level = "high"
    elif score >= 0.20:
        level = "medium"
    else:
        level = "low"
    return {
        "company_num": int(company_num),
        "risk_score": round(score, 2),
        "flagged": score >= THRESHOLD,
        "risk_level": level,
        "missing_fields": int(row[FEATURES].isna().sum().sum()),
    }


# ---------- Tool 3: policy search ----------
def search_policy(question, k=2):
    sims = cosine_similarity(vectorizer.transform([question]), rule_vectors)[0]
    best = sims.argsort()[::-1][:k]
    found = [rules[i] for i in best if sims[i] > 0]
    return found if found else ["No relevant policy found."]