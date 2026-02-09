from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd


TARGET_COLS: List[str] = [
    "Модуль упругости при растяжении, ГПа",
    "Прочность при растяжении, МПа",
]
RECOMMEND_TARGET: str = "Соотношение матрица-наполнитель"


@dataclass(frozen=True)
class IQRBounds:
    lower: float
    upper: float


@dataclass(frozen=True)
class OutlierResult:
    df: pd.DataFrame
    bounds: Dict[str, IQRBounds]
    dropped_rows: int = 0


class OutlierProcessor:

    def __init__(self, k: float = 1.5, exclude_cols: Optional[List[str]] = None) -> None:
        self.k = float(k)
        self.exclude_cols = set(exclude_cols or [])

    def _feature_cols(self, df: pd.DataFrame) -> List[str]:
        numeric = df.select_dtypes(include="number").columns.tolist()
        return [c for c in numeric if c not in self.exclude_cols]

    def compute_bounds(self, df: pd.DataFrame) -> Dict[str, IQRBounds]:
        bounds: Dict[str, IQRBounds] = {}
        for col in self._feature_cols(df):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = float(q1 - self.k * iqr)
            upper = float(q3 + self.k * iqr)
            bounds[col] = IQRBounds(lower=lower, upper=upper)
        return bounds

    def winsorize_iqr(self, df: pd.DataFrame) -> OutlierResult:
        out = df.copy()
        bounds = self.compute_bounds(out)
        for col, b in bounds.items():
            out[col] = out[col].clip(b.lower, b.upper)
        return OutlierResult(df=out, bounds=bounds, dropped_rows=0)

    def drop_rows_iqr(self, df: pd.DataFrame) -> OutlierResult:
        out = df.copy()
        bounds = self.compute_bounds(out)

        mask = pd.Series(True, index=out.index)
        for col, b in bounds.items():
            mask &= out[col].between(b.lower, b.upper)

        dropped = int((~mask).sum())
        out = out.loc[mask].copy()
        return OutlierResult(df=out, bounds=bounds, dropped_rows=dropped)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(3),
        }
    ).sort_values("missing_count", ascending=False)


def stats_report(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    rep = num.describe().T
    rep["median"] = num.median()
    return rep