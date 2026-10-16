import os
from pathlib import Path
import pandas as pd

# Define base directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "placement_data.csv"

def load_data(filepath=None):
    if filepath is None:
        filepath = DATA_PATH
    df = pd.read_csv(filepath)
    return df

def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "placement status"
    }

if __name__ == "__main__":
    df = load_data()
    summary = get_summary(df)
    print("Summary:", summary)


    print("\nFirst 5 rows:")
    print(df.head())
    print(df.tail())

