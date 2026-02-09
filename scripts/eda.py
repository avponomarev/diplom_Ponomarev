import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def safe_filename(name: str) -> str:
    name = name.replace("/", "_")
    name = re.sub(r'[\\:*?"<>|]', "_", name)
    return name.strip()


def load_data(path_bp: str, path_nup: str) -> pd.DataFrame:
    df_bp = pd.read_excel(path_bp, index_col=0)
    df_nup = pd.read_excel(path_nup, index_col=0)
    df = df_bp.join(df_nup, how="inner")
    return df

def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe().T
    desc["median"] = df.median(numeric_only=True)
    return desc


def plot_histograms(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Распределение признака «{col}»")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.tight_layout()

        filename = f"hist_{safe_filename(col)}.png"
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path, dpi=200)
        plt.close()

def plot_boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot признака «{col}»")
        plt.tight_layout()

        filename = f"boxplot_{safe_filename(col)}.png"
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path, dpi=200)
        plt.close()

def plot_pairplot(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    sample_df = numeric_df.sample(min(300, len(numeric_df)), random_state=42)

    sns.pairplot(sample_df, corner=True)
    path = os.path.join(FIGURES_DIR, "pairplot.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_correlation(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Корреляционная матрица признаков")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()

def check_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()

def main():
    data_bp = os.path.join(BASE_DIR, "data", "X_bp.xlsx")
    data_nup = os.path.join(BASE_DIR, "data", "X_nup.xlsx")

    df = load_data(data_bp, data_nup)

    print("Размер датасета:", df.shape)
    print("\nПропуски по столбцам:\n", check_missing(df))

    desc = describe_data(df)
    desc.to_csv(os.path.join(FIGURES_DIR, "describe.csv"))

    plot_histograms(df)
    plot_boxplots(df)
    plot_pairplot(df)
    plot_correlation(df)

    print("EDA завершён. Все графики сохранены в reports/figures")


if __name__ == "__main__":
    main()