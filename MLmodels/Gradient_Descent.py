import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identity_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data,
)


def gradient_descent(X, y, learning_rate, epochs):

    m, n = X.shape
    theta = np.zeros(n)      # weights for each feature
    bias = 0.0
    mse_history = []

    for epoch in range(epochs):
        y_hat = X.dot(theta) + bias
        error = y_hat - y

        mse = np.mean(error ** 2)
        mse_history.append(mse)

        grad_theta = (2 / m) * X.T.dot(error)
        grad_bias = (2 / m) * np.sum(error)

        theta -= learning_rate * grad_theta
        bias -= learning_rate * grad_bias

    return theta, bias, mse_history


def predict(X, theta, bias):
    return X.dot(theta) + bias


def evaluate_model(y_test, y_pred):
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = mse ** 0.5
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def main():
    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",
        drop_columns=["StudentID", "PlacementStatus", "IsAnomaly"],
    )
    print("Train Dataset Shape:", X_train.shape)
    print("Test Dataset Shape:", X_test.shape)

    numerical_features, categorical_features = identity_features(X_train)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    one_hot_features = [
        "Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"
    ]
    ordinal_features = ["CollegeTier", "CGPA_Tier"]

    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    print("Standardization completed.")

    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal encoding completed.")

    # Convert to numpy arrays for gradient descent
    X_train_arr = X_train.values.astype(float)
    X_test_arr = X_test.values.astype(float)
    y_train_arr = y_train.values.astype(float)
    y_test_arr = y_test.values.astype(float)

    theta, bias, mse_history = gradient_descent(
        X_train_arr, y_train_arr, learning_rate=0.01, epochs=1000
    )

    print("Theta:", theta)
    print("Bias:", bias)
    print("Final Train MSE:", mse_history[-1])

    y_pred = predict(X_test_arr, theta, bias)
    metrics = evaluate_model(y_test_arr, y_pred)
    print("Test Metrics:", metrics)

    plt.plot(mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Gradient Descent Convergence")
    plt.show()


if __name__ == "__main__":
    main()