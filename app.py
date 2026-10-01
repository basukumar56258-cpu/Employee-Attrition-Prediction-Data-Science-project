import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "attrition_model.joblib")
DATA_PATH = os.path.join(BASE, "data", "raw", "employee_attrition.csv")

app = Flask(__name__)
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
df = pd.read_csv(DATA_PATH)

def dashboard_stats():
    return {
        "employees": len(df),
        "attrition": int((df["Attrition"] == "Yes").sum()),
        "rate": round((df["Attrition"] == "Yes").mean() * 100, 1),
        "income": round(df["MonthlyIncome"].mean())
    }

@app.route("/")
def index():
    stats = dashboard_stats()
    return render_template("index.html", stats=stats, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", stats=dashboard_stats(),
                               prediction="Model not trained. Run: python src/train_model.py")

    data = {
        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Department": request.form["Department"],
        "JobRole": request.form["JobRole"],
        "Education": request.form["Education"],
        "MaritalStatus": request.form["MaritalStatus"],
        "MonthlyIncome": float(request.form["MonthlyIncome"]),
        "JobLevel": int(request.form["JobLevel"]),
        "YearsAtCompany": int(request.form["YearsAtCompany"]),
        "TotalWorkingYears": int(request.form["TotalWorkingYears"]),
        "JobSatisfaction": int(request.form["JobSatisfaction"]),
        "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"]),
        "WorkLifeBalance": int(request.form["WorkLifeBalance"]),
        "OverTime": request.form["OverTime"],
        "DistanceFromHome": int(request.form["DistanceFromHome"]),
        "YearsSinceLastPromotion": int(request.form["YearsSinceLastPromotion"]),
        "TrainingTimesLastYear": int(request.form["TrainingTimesLastYear"]),
        "PerformanceRating": int(request.form["PerformanceRating"]),
        "StockOptionLevel": int(request.form["StockOptionLevel"]),
        "NumCompaniesWorked": int(request.form["NumCompaniesWorked"]),
        "BusinessTravel": request.form["BusinessTravel"]
    }
    x = pd.DataFrame([data])
    pred = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1]) * 100
    result = f"{'High Attrition Risk' if pred else 'Low Attrition Risk'} — estimated risk: {probability:.1f}%"
    return render_template("index.html", stats=dashboard_stats(), prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
