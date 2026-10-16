from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

def load_preprocessed_data():
    train_path = BASE_DIR / "data" / "preprocessed_train.csv"
    test_path = BASE_DIR / "data" / "preprocessed_test.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def split_features_target(train_data, test_data):
    leakage_cols = [c for c in ["PlacementStatus", "StudentID", "Salary Package", "IsAnomaly"] if c in train_data.columns]
    X_train = train_data.drop(columns=leakage_cols)
    y_train = train_data["PlacementStatus"]
    X_test = test_data.drop(columns=[c for c in leakage_cols if c in test_data.columns])
    y_test = test_data["PlacementStatus"]
    return X_train, y_train, X_test, y_test

def create_model():
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy")
    print(model.score(X_test, y_test))
    print("Classification Report")
    print(classification_report(y_test, y_pred))

def save_model(model):
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logistic_reg.pkl"
    joblib.dump(model, model_path)
    print("Model saved to:", model_path)

if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()
    print("Training Data Shape")
    print(train_data.shape)
    print("Testing Data Type")
    print(test_data.shape)
    X_train, y_train, X_test, y_test = split_features_target(train_data, test_data)
    print("Train Shape")
    print(X_train.shape)
    print("Test Shape")
    print(X_test.shape)
    print("Test Type")
    print(y_test.shape)
    print("Train Shape")
    print(y_train.shape)

    model = create_model()
    print("Logistic Regression Model created")
    model = train_model(model, X_train, y_train)
    print("Logistic Regression Model trained completed")
    evaluate_model(model, X_test, y_test)
    save_model(model)

