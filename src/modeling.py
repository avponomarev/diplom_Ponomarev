from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .metrics import regression_metrics
from .preprocessing import TARGET_COLS


@dataclass(frozen=True)
class TrainResult:
    feature_cols: List[str]
    models: Dict[str, Any]
    best_name: str
    best_model: Any
    metrics: pd.DataFrame
    params: Dict[str, Dict[str, Any]]
    splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]  # Xtr, Xte, ytr, yte


class ModelTrainer:


    def __init__(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        cv_splits: int = 10,
        n_jobs: int = -1,
        scoring: str = "neg_mean_squared_error",
    ) -> None:
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.cv_splits = int(cv_splits)
        self.n_jobs = int(n_jobs)
        self.scoring = scoring

    def fit(self, df: pd.DataFrame) -> TrainResult:
        if any(t not in df.columns for t in TARGET_COLS):
            raise ValueError(f"В df нет таргет-колонок {TARGET_COLS}. Проверь загрузку/merge данных.")

        X = df.drop(columns=TARGET_COLS)
        y = df[TARGET_COLS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        models: Dict[str, Any] = {}
        params: Dict[str, Dict[str, Any]] = {}

        ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(Ridge(random_state=self.random_state))),
            ]
        )
        ridge_grid = {"model__estimator__alpha": [0.1, 1.0, 10.0, 100.0]}
        gs_ridge = GridSearchCV(
            ridge,
            ridge_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        gs_ridge.fit(X_train, y_train)
        models["Ridge"] = gs_ridge.best_estimator_
        params["Ridge"] = gs_ridge.best_params_

        rf = RandomForestRegressor(random_state=self.random_state)
        rf_grid = {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        gs_rf = GridSearchCV(
            rf,
            rf_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        gs_rf.fit(X_train, y_train)
        models["RandomForest"] = gs_rf.best_estimator_
        params["RandomForest"] = gs_rf.best_params_

        gbr = MultiOutputRegressor(GradientBoostingRegressor(random_state=self.random_state))
        gbr_grid = {
            "estimator__n_estimators": [200, 400],
            "estimator__learning_rate": [0.05, 0.1],
            "estimator__max_depth": [2, 3],
        }
        gs_gbr = GridSearchCV(
            gbr,
            gbr_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        gs_gbr.fit(X_train, y_train)
        models["GradientBoosting"] = gs_gbr.best_estimator_
        params["GradientBoosting"] = gs_gbr.best_params_

        rows = []
        for name, m in models.items():
            pred_tr = m.predict(X_train)
            pred_te = m.predict(X_test)

            mae_tr, rmse_tr, r2_tr = regression_metrics(y_train, pred_tr)
            mae_te, rmse_te, r2_te = regression_metrics(y_test, pred_te)

            rows.append(
                {
                    "model": name,
                    "MAE_train_E": mae_tr[0],
                    "MAE_train_sigma": mae_tr[1],
                    "RMSE_train_E": rmse_tr[0],
                    "RMSE_train_sigma": rmse_tr[1],
                    "R2_train_E": r2_tr[0],
                    "R2_train_sigma": r2_tr[1],
                    "MAE_test_E": mae_te[0],
                    "MAE_test_sigma": mae_te[1],
                    "RMSE_test_E": rmse_te[0],
                    "RMSE_test_sigma": rmse_te[1],
                    "R2_test_E": r2_te[0],
                    "R2_test_sigma": r2_te[1],
                }
            )

        metrics_df = pd.DataFrame(rows)
        metrics_df["RMSE_test_avg"] = (metrics_df["RMSE_test_E"] + metrics_df["RMSE_test_sigma"]) / 2
        metrics_df = metrics_df.sort_values("RMSE_test_avg").reset_index(drop=True)

        best_name = str(metrics_df.iloc[0]["model"])
        best_model = models[best_name]

        return TrainResult(
            feature_cols=list(X.columns),
            models=models,
            best_name=best_name,
            best_model=best_model,
            metrics=metrics_df,
            params=params,
            splits=(X_train, X_test, y_train, y_test),
        )

    @staticmethod
    def save_best(
        best_model: Any,
        feature_cols: List[str],
        best_model_path: str,
        meta_path: str,
    ) -> None:
        joblib.dump(best_model, best_model_path)
        meta = {"feature_cols": feature_cols}
        PathLike = __import__("pathlib").Path  # чтобы не тащить лишний импорт сверху
        PathLike(meta_path).write_text(__import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def train_models(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    cv_splits: int = 10,
) -> Dict[str, Any]:

    trainer = ModelTrainer(test_size=test_size, random_state=random_state, cv_splits=cv_splits)
    res = trainer.fit(df)
    return {
        "feature_cols": res.feature_cols,
        "models": res.models,
        "best_name": res.best_name,
        "best_model": res.best_model,
        "metrics": res.metrics,
        "params": res.params,
        "splits": res.splits,
    }