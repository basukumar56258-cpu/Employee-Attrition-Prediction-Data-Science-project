# Employee Attrition Prediction

A professional end-to-end Data Science project that analyzes employee attrition and predicts whether an employee is likely to leave the organization.

## Features
- HR dataset exploration and cleaning
- EDA and business insights
- Attrition KPI dashboard
- Machine Learning prediction using Logistic Regression
- Random Forest comparison
- Interactive web application with Flask
- Single-employee prediction form
- Model metrics and feature importance
- Clean modular project structure

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Flask, Joblib, HTML/CSS

## Run in VS Code

### 1. Open the project
Open this folder in VS Code.

### 2. Create virtual environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python src/train_model.py
```

This creates the trained model in `models/attrition_model.joblib` and evaluation files in `reports/`.

### 5. Start the application
```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

## Project Workflow
1. Load HR data
2. Clean and validate data
3. Perform exploratory data analysis
4. Engineer categorical/numerical features
5. Train Logistic Regression and Random Forest
6. Select the better model using F1 score
7. Save the production pipeline
8. Serve predictions through Flask

## Important
The included dataset is synthetic for demonstration/portfolio purposes. It should not be used for real HR decisions without proper validation, privacy review, fairness testing, and domain oversight.
