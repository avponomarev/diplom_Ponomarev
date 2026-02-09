import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from src.data_loader import load_and_merge
from src.preprocessing import OutlierProcessor, TARGET_COLS, RECOMMEND_TARGET

# Если у тебя в src/recommender_nn.py есть load_recommender — используем.
# Если нет — загрузим модель/скейлер напрямую.
try:
    from src.recommender_nn import load_recommender  # type: ignore
except Exception:  # noqa
    load_recommender = None


DATA_BP = BASE / "data" / "X_bp.xlsx"
DATA_NUP = BASE / "data" / "X_nup.xlsx"

MODEL_PATH = BASE / "models" / "best_model.pkl"
MODEL_META_PATH = BASE / "models" / "best_model_meta.json"

REC_MODEL_PATH = BASE / "models" / "recommender_model.pkl"
REC_SCALER_PATH = BASE / "models" / "recommender_scaler.pkl"


st.set_page_config(page_title="Прогноз свойств композитов", layout="wide")
st.title("Прогнозирование конечных свойств композитов")


def safe_joblib_load(path: Path):
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data
def load_data_cached() -> pd.DataFrame:
    # load_and_merge у тебя работает и с str, и с Path; оставляем Path
    return load_and_merge(DATA_BP, DATA_NUP)


@st.cache_data
def preprocess_for_ui(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Для GUI делаем мягкую обработку выбросов (winsorize) только по фичам.
    # Таргеты исключаем, чтобы их не клипать.
    op = OutlierProcessor(k=1.5, exclude_cols=list(TARGET_COLS) + [RECOMMEND_TARGET])
    res = op.winsorize_iqr(df_raw)
    return res.df


def build_inputs(df: pd.DataFrame, cols_list: list[str], key_prefix: str) -> dict:
    ui_cols = st.columns(3)
    values = {}
    for i, col in enumerate(cols_list):
        c = ui_cols[i % 3]

        if col not in df.columns:
            # если meta содержит колонку, которой нет в df — всё равно дадим ввести
            values[col] = c.number_input(col, value=0.0, format="%.6f", key=f"{key_prefix}_{col}")
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            # Нечисловые — пропускаем (или можно сделать selectbox, но у тебя датасет числовой)
            continue

        mn, mx = float(series.min()), float(series.max())
        default = float(series.median())

        # Streamlit иногда ругается, если min==max — тогда убираем ограничения
        if mn == mx:
            values[col] = c.number_input(col, value=default, format="%.6f", key=f"{key_prefix}_{col}")
        else:
            values[col] = c.number_input(
                col, min_value=mn, max_value=mx, value=default, format="%.6f", key=f"{key_prefix}_{col}"
            )
    return values


# --- load data ---
df_raw = load_data_cached()
df = preprocess_for_ui(df_raw)

# feature columns for default UI (если meta нет)
feature_cols_fallback = [c for c in df.columns if c not in TARGET_COLS and c != RECOMMEND_TARGET]
feature_cols_fallback = df[feature_cols_fallback].select_dtypes(include="number").columns.tolist()

tab1, tab2 = st.tabs(["Прогноз (E и σ)", "Рекомендация соотношения"])


# ============================
# TAB 1: Prediction
# ============================
with tab1:
    st.subheader("Ввод параметров материала")

    # Загружаем meta, чтобы гарантировать порядок признаков
    meta = load_json(MODEL_META_PATH) if MODEL_META_PATH.exists() else None
    if meta and isinstance(meta, dict) and isinstance(meta.get("feature_cols"), list):
        feature_cols = list(meta["feature_cols"])
        model_name = meta.get("best_name", "best_model")
    else:
        feature_cols = feature_cols_fallback
        model_name = "best_model"

    user_input = build_inputs(df, feature_cols, key_prefix="pred")
    X = pd.DataFrame([user_input], columns=feature_cols)

    st.subheader("Модель прогноза")

    if not MODEL_PATH.exists():
        st.error(
            "Файл модели не найден: models/best_model.pkl\n\n"
            "Сначала обучи и сохрани модель командой:\n"
            "python scripts/train_models_run.py"
        )
        st.stop()

    model, err = safe_joblib_load(MODEL_PATH)
    if err:
        st.error(
            "Не удалось загрузить models/best_model.pkl.\n\n"
            f"Ошибка: {err}\n\n"
            "Частая причина — несовместимость версий numpy/sklearn.\n"
            "Решение: переобучи модель в этом же окружении и пересохрани."
        )
        st.stop()

    st.success(f"Загружена модель: **{model_name}**")

    if st.button("Сделать прогноз", type="primary"):
        try:
            pred = np.asarray(model.predict(X)).reshape(1, -1)
            st.markdown("### ✅ Результат прогноза")
            st.metric("Модуль упругости при растяжении, ГПа", f"{pred[0, 0]:.3f}")
            st.metric("Прочность при растяжении, МПа", f"{pred[0, 1]:.3f}")
        except Exception as e:
            st.error(f"Ошибка при прогнозе: {e}")


# ============================
# TAB 2: Recommender
# ============================
with tab2:
    st.subheader("Задайте желаемые свойства + остальные параметры")

    c1, c2 = st.columns(2)
    target_E = c1.number_input("Желаемый модуль упругости (ГПа)", value=float(df[TARGET_COLS[0]].median()))
    target_sigma = c2.number_input("Желаемая прочность (МПа)", value=float(df[TARGET_COLS[1]].median()))

    # Колонки для входа рекомендателя: все числовые, кроме таргета рекомендателя
    # (а целевые E/sigma мы будем подставлять из полей выше)
    rec_input_cols = [c for c in df.columns if c != RECOMMEND_TARGET]
    rec_input_cols = df[rec_input_cols].select_dtypes(include="number").columns.tolist()

    # Построим ввод
    ui_cols = st.columns(3)
    rec_input = {}
    for i, col in enumerate(rec_input_cols):
        if col == TARGET_COLS[0]:
            rec_input[col] = float(target_E)
            continue
        if col == TARGET_COLS[1]:
            rec_input[col] = float(target_sigma)
            continue

        c = ui_cols[i % 3]
        series = df[col]
        mn, mx = float(series.min()), float(series.max())
        default = float(series.median())
        if mn == mx:
            rec_input[col] = c.number_input(col, value=default, format="%.6f", key=f"rec_{col}")
        else:
            rec_input[col] = c.number_input(
                col, min_value=mn, max_value=mx, value=default, format="%.6f", key=f"rec_{col}"
            )

    x_rec = pd.DataFrame([rec_input])

    st.subheader("Нейросеть-рекомендатель")

    if not (REC_MODEL_PATH.exists() and REC_SCALER_PATH.exists()):
        st.warning(
            "Модель рекомендаций не найдена.\n\n"
            "Нужно, чтобы существовали файлы:\n"
            "- models/recommender_model.pkl\n"
            "- models/recommender_scaler.pkl\n\n"
            "Сначала обучи рекомендатель отдельным скриптом."
        )
    else:
        # Загрузка рекомендателя: либо через load_recommender, либо напрямую joblib
        if load_recommender is not None:
            try:
                model_rec, scaler_rec, cols_rec = load_recommender(str(REC_MODEL_PATH), str(REC_SCALER_PATH))
            except Exception as e:
                st.error(f"Не удалось загрузить рекомендатель через load_recommender: {e}")
                st.stop()
        else:
            model_rec, err1 = safe_joblib_load(REC_MODEL_PATH)
            scaler_rec, err2 = safe_joblib_load(REC_SCALER_PATH)
            if err1 or err2:
                st.error(f"Ошибка загрузки рекомендателя: {err1 or ''} {err2 or ''}".strip())
                st.stop()
            # В этом фолбэке считаем, что рекомендатель обучался на тех же колонках
            cols_rec = list(x_rec.columns)

        # кнопка расчёта
        if st.button("Рекомендовать соотношение матрица–наполнитель", type="primary"):
            try:
                X_in = x_rec[cols_rec].values
                Xs = scaler_rec.transform(X_in)
                y_pred = float(np.asarray(model_rec.predict(Xs)).reshape(-1)[0])
                st.success(f"✅ Рекомендуемое соотношение матрица–наполнитель: **{y_pred:.4f}**")
            except Exception as e:
                st.error(f"Ошибка при расчёте рекомендации: {e}")


st.divider()
st.caption("Дипломный проект: прогнозирование конечных свойств композиционных материалов. Автор — Пономарев А.В.")