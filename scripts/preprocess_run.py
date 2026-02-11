import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader import load_and_merge
from src.preprocessing import winsorize_iqr, TARGET_COLS
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
DATA_BP = BASE / "data" / "X_bp.xlsx"
DATA_NUP = BASE / "data" / "X_nup.xlsx"

FIG_DIR = BASE / "reports" / "figures"
TAB_DIR = BASE / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:

    name = name.strip()
    name = re.sub(r"[\/\\\:\*\?\"<>\|]", "_", name)
    name = re.sub(r"[,\s]+", "_", name)
    name = re.sub(r"__+", "_", name)         
    return name


def save_hist_before_after(df_before, df_after, col, out_path):
    """Гистограммы до/после нормализации для одного признака"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(df_before[col], bins=30)
    plt.title(f"{col} (до нормализации)")

    plt.subplot(1, 2, 2)
    plt.hist(df_after[col], bins=30)
    plt.title(f"{col} (после нормализации)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_boxplot(df, col, out_path, title):
    """Boxplot одного признака"""
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    print("Загрузка и объединение данных...")
    df_raw = load_and_merge(str(DATA_BP), str(DATA_NUP))
    print("Размер данных:", df_raw.shape)

    # --- 1) Проверка пропусков ---
    na_counts = df_raw.isna().sum().sort_values(ascending=False)
    na_counts.to_csv(TAB_DIR / "missing_values.csv")
    print("Пропуски сохранены: reports/tables/missing_values.csv")

    # --- 2) Статистика mean/median/min/max ---
    stats = df_raw.describe().T
    stats["median"] = df_raw.median(numeric_only=True)
    stats = stats[["mean", "median", "std", "min", "max"]]
    stats.to_csv(TAB_DIR / "stats_before.csv")
    print("Статистика ДО сохранена: reports/tables/stats_before.csv")

    # --- 3) Обработка выбросов (winsorization IQR) ---
    print("Обработка выбросов (IQR winsorization)...")
    df_clean, bounds = winsorize_iqr(df_raw, k=1.5)
    pd.DataFrame(bounds).T.to_csv(TAB_DIR / "iqr_bounds.csv")
    print("Границы IQR сохранены: reports/tables/iqr_bounds.csv")

    # Статистика после очистки выбросов
    stats_clean = df_clean.describe().T
    stats_clean["median"] = df_clean.median(numeric_only=True)
    stats_clean = stats_clean[["mean", "median", "std", "min", "max"]]
    stats_clean.to_csv(TAB_DIR / "stats_after_outliers.csv")
    print("Статистика ПОСЛЕ очистки выбросов сохранена: reports/tables/stats_after_outliers.csv")

    # --- 4) Нормализация признаков (StandardScaler) ---
    print("Нормализация признаков StandardScaler...")
    feature_cols = [c for c in df_clean.columns if c not in TARGET_COLS]
    X = df_clean[feature_cols]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # min/max до и после
    minmax_before = pd.DataFrame({"min": X.min(), "max": X.max()})
    minmax_after = pd.DataFrame({"min": X_scaled.min(), "max": X_scaled.max()})
    minmax_before.to_csv(TAB_DIR / "minmax_before_scaling.csv")
    minmax_after.to_csv(TAB_DIR / "minmax_after_scaling.csv")
    print("Min/Max ДО и ПОСЛЕ сохранены: reports/tables/minmax_before_scaling.csv и minmax_after_scaling.csv")

    print("Сохранение гистограмм ДО/ПОСЛЕ нормализации...")
    for col in feature_cols:
        safe_col = sanitize_filename(col)
        out_path = FIG_DIR / f"hist_before_after_{safe_col}.png"
        save_hist_before_after(X, X_scaled, col, out_path)

    print("Гистограммы сохранены в reports/figures/")

    print("Сохранение boxplot ДО и ПОСЛЕ очистки выбросов...")
    for col in feature_cols:
        safe_col = sanitize_filename(col)
        save_boxplot(df_raw, col, FIG_DIR / f"box_before_{safe_col}.png", f"{col} (до очистки)")
        save_boxplot(df_clean, col, FIG_DIR / f"box_after_{safe_col}.png", f"{col} (после очистки)")

    print("Boxplot сохранены в reports/figures/")

    print("Сохранение корреляционной матрицы...")
    corr = df_clean.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Корреляционная матрица (после очистки)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=250)
    plt.close()

    print("Корреляционная матрица сохранена: reports/figures/correlation_matrix.png")

    print("Таблицы → reports/tables/")
    print("Графики → reports/figures/")


if __name__ == "__main__":
    main()
