import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "raw", "employee_attrition.csv"))
report = os.path.join(ROOT, "reports")
os.makedirs(report, exist_ok=True)

# Attrition by department
plt.figure(figsize=(9,5))
sns.countplot(data=df, x="Department", hue="Attrition")
plt.xticks(rotation=25, ha="right")
plt.title("Attrition by Department")
plt.tight_layout()
plt.savefig(os.path.join(report, "attrition_by_department.png"), dpi=150)
plt.close()

# Monthly income distribution
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
plt.title("Monthly Income vs Attrition")
plt.tight_layout()
plt.savefig(os.path.join(report, "income_vs_attrition.png"), dpi=150)
plt.close()

print("EDA reports generated in reports/")
