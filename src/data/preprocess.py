import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.data.load_data import load_data, BASE_DIR   # your custom loader

def split_data(df,target_column,drop_columns=None,stratify=False):
    # Target column should match your dataset
    if drop_columns is None:
        drop_columns = []
    X = df.drop(columns=drop_columns + [target_column])
    y = df[target_column]
    stratify_value = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_value
    )
    return X_train, X_test, y_train, y_test

def split_X_data(df, drop_columns=None):
    # Unsupervised version of split_data: no target column, no train/test split
    if drop_columns is None:
        drop_columns = []
    X = df.drop(columns=drop_columns)
    return X

def identity_features(X):
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return numerical_features, categorical_features

def handle_missing_values(X_train, X_test, numerical_features):
    imputer = SimpleImputer(strategy="median")
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_features] = imputer.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = imputer.transform(X_test[numerical_features])

    return X_train, X_test, imputer

def standardize_data(X_train, X_test, numerical_features):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train, X_test, scaler

def one_hot_encode_data(X_train, X_test, one_hot_features):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_encoded = encoder.fit_transform(X_train[one_hot_features])
    test_encoded = encoder.transform(X_test[one_hot_features])

    encoded_columns = encoder.get_feature_names_out(one_hot_features)

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns, index=X_train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_columns, index=X_test.index)

    # Remove only one-hot features
    X_train = X_train.drop(columns=one_hot_features)
    X_test = X_test.drop(columns=one_hot_features)

    # Add encoded columns
    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)

    return X_train, X_test, encoder

def ordinal_encode_data(X_train, X_test, ordinal_features):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_encoded = encoder.fit_transform(X_train[ordinal_features])
    test_encoded = encoder.transform(X_test[ordinal_features])

    train_encoded_df = pd.DataFrame(train_encoded, columns=ordinal_features, index=X_train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=ordinal_features, index=X_test.index)

    # Remove only ordinal features
    X_train = X_train.drop(columns=ordinal_features)
    X_test = X_test.drop(columns=ordinal_features)

    # Add encoded columns
    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)

    return X_train, X_test, encoder

if __name__ == "__main__":
    df = load_data()
    print("Original Data Shape:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df, target_column="PlacementStatus")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    numerical_features, categorical_features = identity_features(X_train)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    # Define which features to one-hot encode and which to ordinal encode
    one_hot_features = [
        'Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs'
    ]
    ordinal_features = [
        'CollegeTier', 'CGPA_Tier'
    ]

    # Handle missing values
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    # Standardize numerical features
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    print("Standardization completed.")

    # One-hot encode categorical features
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    # Ordinal encode ordered categorical features
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal encoding completed.")

    # Add target back for saving
    X_train["PlacementStatus"] = y_train
    X_test["PlacementStatus"] = y_test

    # Save preprocessed data
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(data_dir / "preprocessed_train.csv", index=False)
    X_test.to_csv(data_dir / "preprocessed_test.csv", index=False)

    print("Preprocessing finished. Train and test CSVs saved.")
    print(X_train.head())
    print(X_test.head())