import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocess import (split_data,identity_features,handle_missing_values,one_hot_encode_data,ordinal_encode_data)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def create_model():
    model = RandomForestClassifier(
        n_estimators=100,max_features='sqrt',random_state=42,oob_score=True
    )
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)

    print("Accuracy", accuracy)
    print("OOB Score", model.oob_score_)

def main():
    df = load_data()
    print("Original Dataset Shape:")
    print(df.shape)
    X_train, X_test, y_train, y_test = split_data(df, target_column="PlacementStatus", drop_columns=[
        "StudentID",
        "Salary Package",
        "IsAnomaly"
    ])
    print("Trained DataSet Shape:")
    print(X_train.shape)
    print("Test DataSet Shape:")
    print(X_test.shape)

    numerical_features, categorical_features = (identity_features(X_train))
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    one_hot_features = [
    'Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs'
    ]
    ordinal_features = [
    'CollegeTier', 'CGPA_Tier'
    ]
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    # Ordinal encode ordered categorical features
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal encoding completed.")

    model = create_model()
    model = train_model(model,X_train,y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()