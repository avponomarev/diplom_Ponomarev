from __future__ import annotations

import sys
from pathlib import Path
import json

import joblib

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.data_loader import load_and_merge
from src.modeling import ModelTrainer


def main() -> None:
    data_bp = BASE / "data" / "X_bp.xlsx"
    data_nup = BASE / "data" / "X_nup.xlsx"

    df = load_and_merge(data_bp, data_nup)

    trainer = ModelTrainer(test_size=0.3, random_state=42, cv_splits=10)
    result = trainer.fit(df)

    models_dir = BASE / "models"
    reports_dir = BASE / "reports" / "tables"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pkl"
    meta_path = models_dir / "best_model_meta.json"
    metrics_path = reports_dir / "metrics.csv"
    params_path = reports_dir / "best_params.json"

    joblib.dump(result.best_model, best_model_path)
    meta_path.write_text(
        json.dumps({"feature_cols": result.feature_cols, "best_name": result.best_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    params_path.write_text(json.dumps(result.params, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Training models complete.")
    print(f"✅ Best model: {result.best_name}")
    print("✅ Saved:")
    print(" -", best_model_path)
    print(" -", meta_path)
    print(" -", metrics_path)
    print(" -", params_path)
    print("\nTop metrics (sorted by RMSE_test_avg):")
    print(result.metrics.head(5).to_string(index=False))


if __name__ == "__main__":
    main()