import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import seaborn as sns
from src.data.load_data import load_data, BASE_DIR
from matplotlib import pyplot as plt

RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = BASE_DIR / "app" / "static" / "charts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(filename):
    plt.savefig(RESULTS_DIR / filename, bbox_inches='tight')
    plt.savefig(CHARTS_DIR / filename, bbox_inches='tight')

def basic_eda(df):
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nRows 25 to 35:")
    print(df.iloc[25:36])

    print("\nSample of 10 records:")
    print(df.sample(10))

    print("\nColumn names:")
    print(df.columns)

    print("\nDatatypes of features:")
    print(df.dtypes)

    print("\nComplete information:")
    df.info()

    print("\nStatistical measures:")
    print(df.describe())

    print("\nNull values per column:")
    missing = df.isnull().sum()
    print(missing)
    print("\nColumns with missing values:")
    print(missing[missing > 0])

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nTarget variable statistics:")
    print(df["PlacementStatus"].value_counts())

    # Plot distribution of Placement Status
    count = df["PlacementStatus"].value_counts()
    plt.figure(figsize=(6,5))
    plt.bar(count.index, count.values, color="skyblue", edgecolor="black")
    plt.title("Distribution of Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    _save_fig("placement_status_bar.png")
    plt.close()


def univariate_eda(df):
    # Histogram of CGPA
    plt.figure(figsize=(6,5))
    plt.hist(df["CGPA"], bins=10, color="skyblue", edgecolor="black")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    _save_fig("cgpa_hist.png")
    plt.close()

    # Pie chart of Gender
    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6,5))
    plt.pie(gendercount.values, labels=gendercount.index, autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of Gender")
    _save_fig("gender_distribution_pie.png")
    plt.close()


def bivariate_eda(df):
    # Scatter plot CGPA vs Aptitude Test Score
    plt.figure(figsize=(6,5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"], color="skyblue", edgecolor="black")
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    _save_fig("cgpa_vs_aptitude.png")
    plt.close()

    # Boxplot CGPA vs Placement Status
    placed = df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed = df[df["PlacementStatus"] == 0]["CGPA"]
    plt.figure(figsize=(6,5))
    plt.boxplot([placed, not_placed], labels=["Placed", "Not Placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    _save_fig("cgpa_vs_placement.png")
    plt.close()

    # Gender vs Placement Status (Bar chart)
    count = pd.crosstab(df["Gender"], df["PlacementStatus"])
    count.plot(kind="bar", figsize=(6,5), color=["skyblue","lightgreen"])
    plt.title("Gender vs Placement Status")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    _save_fig("gender_vs_placement.png")
    plt.close()


def multivariate_eda(df):

    # Compute correlation only on numeric subset
    correlation = df.corr(numeric_only=True)

    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    _save_fig("correlation_matrix_1.png")
    _save_fig("correlation_matrix.png")
    plt.close()


if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate_eda(df)
    bivariate_eda(df)
    multivariate_eda(df)
