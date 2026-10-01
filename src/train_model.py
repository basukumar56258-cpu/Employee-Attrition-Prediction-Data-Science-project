import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "employee_attrition.csv")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
target = "Attrition"
X = df.drop(columns=[target, "EmployeeNumber"])
y = (df[target] == "Yes").astype(int)

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced", max_depth=10
    )
}

results = {}
best_name, best_pipe, best_f1 = None, None, -1

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4)
    }
    results[name] = metrics
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_name, best_pipe = name, pipe

joblib.dump(best_pipe, os.path.join(MODEL_DIR, "attrition_model.joblib"))

with open(os.path.join(REPORT_DIR, "model_metrics.json"), "w") as f:
    json.dump({"best_model": best_name, "models": results}, f, indent=2)

# Confusion matrix
pred = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Dataset summary
summary = {
    "total_employees": int(len(df)),
    "attrition_count": int((df[target] == "Yes").sum()),
    "attrition_rate": round(float((df[target] == "Yes").mean() * 100), 2),
    "average_income": round(float(df["MonthlyIncome"].mean()), 2)
}
with open(os.path.join(REPORT_DIR, "dataset_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Best model: {best_name}")
print(json.dumps(results, indent=2))
print("Model saved to models/attrition_model.joblib")
