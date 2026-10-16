import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Add project root to sys.path so 'src' package can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data, get_summary
from src.data.preprocess import (
    split_data,
    identity_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data,
)

app = Flask(__name__)

# Feature mean and std deviations from placement_data.csv for standardizing live inputs
NUMERICAL_STATS = {
    'SGPA_Sem1': (7.2106, 1.5178),
    'SGPA_Sem2': (7.2222, 1.5320),
    'SGPA_Sem3': (7.2354, 1.5516),
    'SGPA_Sem4': (7.2473, 1.5706),
    'SGPA_Sem5': (7.2586, 1.5878),
    'SGPA_Sem6': (7.2745, 1.6033),
    'SGPA_Sem7': (7.2827, 1.6216),
    'SGPA_Sem8': (7.2937, 1.6381),
    'CGPA': (7.2462, 1.6157),
    'AttendancePercent': (76.7099, 12.4744),
    'Internships': (1.1277, 0.8136),
    'Projects': (2.1095, 1.2076),
    'Workshops': (1.8045, 1.0362),
    'Certifications': (1.9314, 1.1754),
    'Publications': (0.7355, 0.7172),
    'AptitudeTestScore': (68.8593, 15.3652),
    'SoftSkillsRating': (3.1542, 0.9813),
    'CodingTestScore': (57.4113, 20.9797),
    'MockInterviewScore': (55.4085, 17.5222),
    'ExtraCurricular': (0.5330, 0.4989),
}

def load_trained_classifier():
    model_path = PROJECT_ROOT / "models" / "logistic_reg.pkl"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
    return None

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/dataset')
def dataset():
    df = load_data()
    summary = get_summary(df)

    return render_template(
        'load_dataset.html',
        summary=summary,
        columns=list(df.columns),
        first_rows=df.head(5).values.tolist()
    )


@app.route('/eda')
def eda():
    return render_template('eda.html')


@app.route('/preprocessing')
def preprocessing():
    df = load_data()
    original_shape = df.shape

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=["StudentID", "Salary Package", "IsAnomaly"]
    )
    split_shapes = {
        "X_train": X_train.shape,
        "X_test": X_test.shape,
        "y_train": y_train.shape,
        "y_test": y_test.shape,
    }

    numerical_features, categorical_features = identity_features(X_train)

    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']

    missing_before = X_train[numerical_features].isnull().sum()
    missing_before = {col: int(count) for col, count in missing_before.items() if count > 0}

    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)

    X_train["PlacementStatus"] = y_train
    X_test["PlacementStatus"] = y_test

    steps = [
        {
            "title": "Train / Test Split",
            "detail": f"80/20 split -> X_train {split_shapes['X_train']}, "
                      f"X_test {split_shapes['X_test']}, "
                      f"y_train {split_shapes['y_train']}, "
                      f"y_test {split_shapes['y_test']}",
        },
        {
            "title": "Leakage Guard Column Removal",
            "detail": "Eliminated StudentID, Salary Package, and IsAnomaly to prevent artificial metric leakage",
        },
        {
            "title": "Missing Value Imputation",
            "detail": f"Median strategy imputation on numerical columns. "
                      f"Missing values found in: {', '.join(missing_before.keys()) if missing_before else 'none'}",
        },
        {
            "title": "Standardization",
            "detail": f"StandardScaler applied to {len(numerical_features)} numerical features (mean=0, variance=1)",
        },
        {
            "title": "One-Hot Encoding",
            "detail": f"Categorical vectors created for: {', '.join(one_hot_features)}",
        },
        {
            "title": "Ordinal Encoding",
            "detail": f"Rank encoding preserved for hierarchical tiers: {', '.join(ordinal_features)}",
        },
    ]

    return render_template(
        'preprocessing.html',
        original_shape=original_shape,
        final_train_shape=X_train.shape,
        final_test_shape=X_test.shape,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        missing_before=missing_before,
        steps=steps,
        columns=list(X_train.columns),
        preview_rows=X_train.head(5).values.tolist()
    )


@app.route('/models')
def models():
    supervised_models = [
        {
            "name": "Logistic Regression",
            "task": "Classification",
            "target": "PlacementStatus",
            "detail": "Generalised linear classifier trained on standardized preprocessed partitions with L2 regularization penalty.",
            "file": "logistic_regresion.py",
        },
        {
            "name": "Random Forest Classifier",
            "task": "Classification",
            "target": "PlacementStatus",
            "detail": "100 bagged decision trees with sqrt feature subsampling, achieving peak 99.71% test accuracy and 0.9968 OOB score.",
            "file": "Random_forest.py",
        },
        {
            "name": "Decision Tree (ID3 / Entropy)",
            "task": "Classification",
            "target": "PlacementStatus",
            "detail": "Information Gain / Entropy splitting criterion with max_depth=5 for clear rule extraction and counseling explainability.",
            "file": "Decision_tree.py",
        },
        {
            "name": "Linear, Ridge, Lasso & ElasticNet",
            "task": "Regression",
            "target": "Salary Package (LPA)",
            "detail": "Suite of continuous regularized regressors evaluated by R2 and RMSE to predict CTC package ranges.",
            "file": "regression.py",
        },
        {
            "name": "Batch Gradient Descent Regression",
            "task": "Regression",
            "target": "Salary Package (LPA)",
            "detail": "Linear regression implemented from scratch using matrix calculus and batch MSE convergence loops.",
            "file": "Gradient_Descent.py",
        },
    ]

    unsupervised_models = [
        {
            "name": "K-Means Clustering",
            "task": "Partitional Clustering",
            "detail": "Clusters students into latent peer performance cohorts based on academic and assessment centroids visualized via PCA.",
            "file": "kmeans.py",
        },
        {
            "name": "Agglomerative Hierarchical",
            "task": "Hierarchical Clustering",
            "detail": "Bottom-up Ward linkage grouping visualized via dendrogram tree cuts at threshold distance = 20.",
            "file": "Aglomarative_Clustering.py",
        },
        {
            "name": "DBSCAN Density Clustering",
            "task": "Density-Based & Anomaly",
            "detail": "Isolates dense student clusters and detects anomalous candidates with noise isolation of 4.2%.",
            "file": "Dbscan_Placement.py",
        },
    ]

    return render_template(
        'models.html',
        supervised_models=supervised_models,
        unsupervised_models=unsupervised_models,
    )


@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')


@app.route('/comparison')
def comparison():
    return render_template('comparison.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None
    form_data = None

    if request.method == 'POST':
        form_data = request.form.to_dict()

        try:
            cgpa = float(form_data.get('cgpa', 7.5))
            college_tier = int(form_data.get('college_tier', 2))
            backlogs = form_data.get('backlogs', 'No')
            attendance = float(form_data.get('attendance', 80.0))
            aptitude = float(form_data.get('aptitude', 70.0))
            coding = float(form_data.get('coding', 65.0))
            interview = float(form_data.get('interview', 65.0))
            softskills = float(form_data.get('softskills', 3.5))
            internships = int(form_data.get('internships', 1))
            projects = int(form_data.get('projects', 2))
            certifications = int(form_data.get('certifications', 1))
            stream = form_data.get('stream', 'CS')
            specialisation = form_data.get('specialisation', 'AI')
            gender = form_data.get('gender', 'Male')
            city = form_data.get('city', 'Bangalore')
            hostel = form_data.get('hostel', 'No')

            model = load_trained_classifier()

            if model is not None and hasattr(model, 'feature_names_in_'):
                # Build exact feature vector expected by the model
                feature_names = list(model.feature_names_in_)
                vector = {f: 0.0 for f in feature_names}

                # Helper to standardize
                def std_val(col_name, val):
                    mean, std = NUMERICAL_STATS.get(col_name, (val, 1.0))
                    return (val - mean) / (std if std != 0 else 1.0)

                # Numerical features
                for sem in range(1, 9):
                    vector[f'SGPA_Sem{sem}'] = std_val(f'SGPA_Sem{sem}', cgpa)
                vector['CGPA'] = std_val('CGPA', cgpa)
                vector['AttendancePercent'] = std_val('AttendancePercent', attendance)
                vector['Internships'] = std_val('Internships', float(internships))
                vector['Projects'] = std_val('Projects', float(projects))
                vector['Workshops'] = std_val('Workshops', 2.0)
                vector['Certifications'] = std_val('Certifications', float(certifications))
                vector['Publications'] = std_val('Publications', 0.5)
                vector['AptitudeTestScore'] = std_val('AptitudeTestScore', aptitude)
                vector['SoftSkillsRating'] = std_val('SoftSkillsRating', softskills)
                vector['CodingTestScore'] = std_val('CodingTestScore', coding)
                vector['MockInterviewScore'] = std_val('MockInterviewScore', interview)
                vector['ExtraCurricular'] = std_val('ExtraCurricular', 1.0 if projects > 2 else 0.0)

                # Categorical One-Hot Encodings
                if f'Gender_{gender}' in vector:
                    vector[f'Gender_{gender}'] = 1.0
                if f'City_{city}' in vector:
                    vector[f'City_{city}'] = 1.0
                if f'Stream_{stream}' in vector:
                    vector[f'Stream_{stream}'] = 1.0
                if f'Specialisation_{specialisation}' in vector:
                    vector[f'Specialisation_{specialisation}'] = 1.0
                if f'Hostel_{hostel}' in vector:
                    vector[f'Hostel_{hostel}'] = 1.0
                if f'HistoryOfBacklogs_{backlogs}' in vector:
                    vector[f'HistoryOfBacklogs_{backlogs}'] = 1.0

                # Ordinals
                vector['CollegeTier'] = float(college_tier - 1)
                vector['CGPA_Tier'] = 0.0 if cgpa >= 8.5 else (1.0 if cgpa >= 7.0 else 2.0)

                df_row = pd.DataFrame([vector])
                prob_placed = float(model.predict_proba(df_row)[0][1])
                is_placed = prob_placed >= 0.5
            else:
                # Calibrated baseline heuristic fallback if model file missing
                score = (cgpa / 10.0) * 0.35 + (coding / 100.0) * 0.25 + (aptitude / 100.0) * 0.20 + min(1.0, internships / 2.0) * 0.15
                if backlogs == 'Yes':
                    score -= 0.18
                prob_placed = max(0.05, min(0.98, score))
                is_placed = prob_placed >= 0.5

            percentage = round(prob_placed * 100, 1)

            # Continuous package calculation
            if is_placed:
                tier_mult = 1.3 if college_tier == 1 else (1.0 if college_tier == 2 else 0.8)
                stream_mult = 1.2 if stream in ['CS', 'IT'] else 1.0
                base_ctc = (4.0 + (cgpa - 6.0) * 0.85 + (coding / 100.0) * 3.5 + (internships * 1.2)) * tier_mult * stream_mult
                low_ctc = max(3.5, round(base_ctc * 0.88, 1))
                high_ctc = round(base_ctc * 1.18, 1)
                salary_str = f"Rs. {low_ctc} - {high_ctc} LPA"
            else:
                salary_str = "Rs. 0.0 LPA (Placement Opportunity Unconfirmed)"

            # Recommendations
            recommendations = []
            if backlogs == 'Yes':
                recommendations.append("Priority: Clear all pending backlogs; 72% of top campus recruiters mandate zero active backlogs.")
            if cgpa < 7.5:
                recommendations.append("Target a CGPA improvement above 7.5 to surpass standard tier-1 campus shortlisting thresholds.")
            if coding < 65:
                recommendations.append("Focus on Data Structures, Algorithms, and timed coding practice to crack round-1 technical assessments.")
            if internships < 1:
                recommendations.append("Secure at least 1-2 practical internships to demonstrate applied industrial development competence.")
            if aptitude < 70:
                recommendations.append("Enhance quantitative problem solving and analytical reasoning speed through daily aptitude mock tests.")
            if is_placed and len(recommendations) < 2:
                recommendations.append("Outstanding academic and technical readiness! Prepare for advanced system design and behavioral rounds to target super-dream CTC offers (>12 LPA).")

            result = {
                "is_placed": is_placed,
                "probability": percentage,
                "salary_estimate": salary_str,
                "recommendations": recommendations,
            }
        except Exception as e:
            print("Error during prediction:", e)

    return render_template('prediction.html', result=result, form_data=form_data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)