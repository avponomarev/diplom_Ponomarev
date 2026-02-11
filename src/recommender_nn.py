from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .preprocessing import RECOMMEND_TARGET


def build_dataset(df: pd.DataFrame):

    X = df.drop(columns=[RECOMMEND_TARGET])
    y = df[RECOMMEND_TARGET].values
    return X, y


def train_recommender(
        df: pd.DataFrame,
        model_path: str,
        scaler_path: str,
        test_size: float = 0.3,
        random_state: int = 42
):

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    X, y = build_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values)
    X_test_s = scaler.transform(X_test.values)


    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.2
    )

    print("Training neural network (sklearn MLPRegressor) started...")
    model.fit(X_train_s, y_train)
    print("Training finished!")

    pred_train = model.predict(X_train_s)
    pred_test = model.predict(X_test_s)

    mae_train = mean_absolute_error(y_train, pred_train)
    rmse_train = (mean_squared_error(y_train, pred_train)) ** 0.5

    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = (mean_squared_error(y_test, pred_test)) ** 0.5

    metrics_dict = {
        "MAE_train": float(mae_train),
        "RMSE_train": float(rmse_train),
        "MAE_test": float(mae_test),
        "RMSE_test": float(rmse_test),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "architecture": "Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Linear)"
    }

    print("Metrics:")
    print(f"Train MAE:  {mae_train:.6f} | Train RMSE: {rmse_train:.6f}")
    print(f"Test  MAE:  {mae_test:.6f} | Test  RMSE: {rmse_test:.6f}")

    joblib.dump(model, model_path)
    joblib.dump((scaler, list(X.columns)), scaler_path)

    print("Saved files:")
    print("Model :", model_path, "| exists:", Path(model_path).exists())
    print("Scaler:", scaler_path, "| exists:", Path(scaler_path).exists())

    return metrics_dict, list(X.columns)


def load_recommender(model_path: str, scaler_path: str):

    model = joblib.load(model_path)
    scaler, cols = joblib.load(scaler_path)
    return model, scaler, cols
