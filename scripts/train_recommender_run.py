from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))
from src.data_loader import load_and_merge
from src.recommender_nn import train_recommender


def main() -> None:
    data_bp = BASE / "data" / "X_bp.xlsx"
    data_nup = BASE / "data" / "X_nup.xlsx"
    models_dir = BASE / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "recommender_model.pkl"
    scaler_path = models_dir / "recommender_scaler.pkl"

    df = load_and_merge(data_bp, data_nup)

    result = train_recommender(df, str(model_path), str(scaler_path))

    print("✅ Recommender training complete.")
    print("Result:", result)


if __name__ == "__main__":
    main()