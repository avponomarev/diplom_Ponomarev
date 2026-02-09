from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_and_merge(path_bp: str | Path, path_nup: str | Path) -> pd.DataFrame:

    path_bp = Path(path_bp)
    path_nup = Path(path_nup)

    df_bp = pd.read_excel(path_bp, index_col=0)
    df_nup = pd.read_excel(path_nup, index_col=0)

    df = df_bp.join(df_nup, how="inner")

    if df.empty:
        raise ValueError(
            "После INNER-объединения по индексу датасет пустой. "
            "Проверь, что индексы в X_bp и X_nup совпадают."
        )

    if df.index.hasnans:
        raise ValueError("В индексе есть NaN — это недопустимо для объединения по индексу.")

    return df