from __future__ import annotations

import os
import re
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
APP_DIR = ROOT / "app"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DATA_BP = DATA_DIR / "X_bp.xlsx"
DATA_NUP = DATA_DIR / "X_nup.xlsx"

@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""
    severity: str = "ERROR"


def _print_result(r: CheckResult) -> None:
    status = "✅ PASS" if r.ok else ("⚠️ WARN" if r.severity == "WARN" else "❌ FAIL")
    print(f"{status}  {r.name}")
    if r.details:
        for line in r.details.strip().splitlines():
            print(f"    {line}")


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")

def load_and_merge_inner_by_index(path_bp: Path, path_nup: Path) -> pd.DataFrame:
    df_bp = pd.read_excel(path_bp, index_col=0)
    df_nup = pd.read_excel(path_nup, index_col=0)
    df = df_bp.join(df_nup, how="inner")
    return df


def check_data_files_exist() -> CheckResult:
    missing = []
    if not DATA_BP.exists():
        missing.append(str(DATA_BP))
    if not DATA_NUP.exists():
        missing.append(str(DATA_NUP))
    if missing:
        return CheckResult(
            name="Наличие исходных Excel-файлов",
            ok=False,
            details="Не найдены файлы:\n" + "\n".join(missing),
            severity="ERROR",
        )
    return CheckResult(
        name="Наличие исходных Excel-файлов",
        ok=True,
        details=f"Найдены: {DATA_BP.name}, {DATA_NUP.name}",
        severity="INFO",
    )


def check_merge_inner_index(df: pd.DataFrame) -> CheckResult:
    if df.empty:
        return CheckResult(
            name="Объединение INNER по индексу",
            ok=False,
            details="Результат объединения пустой. Проверь, что индексы в X_bp и X_nup совпадают.",
            severity="ERROR",
        )

    if df.index.isna().any():
        return CheckResult(
            name="Объединение INNER по индексу",
            ok=False,
            details="В индексе есть NaN. Это недопустимо для merge по индексу.",
            severity="ERROR",
        )
    if not df.index.is_unique:
        return CheckResult(
            name="Объединение INNER по индексу",
            ok=False,
            details="Индекс не уникален после объединения. Это может ломать обучение и метрики.",
            severity="ERROR",
        )
    return CheckResult(
        name="Объединение INNER по индексу",
        ok=True,
        details=f"Размер объединённого датасета: {df.shape[0]} строк × {df.shape[1]} столбцов",
        severity="INFO",
    )


def check_missing_values(df: pd.DataFrame) -> CheckResult:
    na = df.isna().sum()
    total_na = int(na.sum())
    if total_na == 0:
        return CheckResult(
            name="Пропуски в данных",
            ok=True,
            details="Пропусков не обнаружено.",
            severity="INFO",
        )

    top = na[na > 0].sort_values(ascending=False).head(10)
    return CheckResult(
        name="Пропуски в данных",
        ok=False,
        details=(
            f"Найдено пропусков: {total_na}\n"
            "Топ столбцов с пропусками:\n" + "\n".join([f"{k}: {int(v)}" for k, v in top.items()])
        ),
        severity="WARN",
    )


TARGET_HINTS = [
    "модуль упругости", "прочность", "растяж", "E", "σ", "sigma",
    "модуль", "упругост", "прочност"
]
RECOMMEND_HINTS = [
    "соотнош", "матриц", "наполн", "отношение", "доля"
]


def guess_target_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str)]
    targets = []
    for c in cols:
        low = c.lower()
        if any(h in low for h in TARGET_HINTS):
            targets.append(c)
    return sorted(set(targets))


def guess_recommend_column(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns if isinstance(c, str)]
    candidates = []
    for c in cols:
        low = c.lower()
        if any(h in low for h in RECOMMEND_HINTS):
            candidates.append(c)
    if not candidates:
        return None
    for key in ["соотнош", "отношение"]:
        for c in candidates:
            if key in c.lower():
                return c
    return candidates[0]


def check_targets_not_in_features(df: pd.DataFrame) -> CheckResult:
    targets = guess_target_columns(df)
    rec = guess_recommend_column(df)

    details_lines = []
    if targets:
        details_lines.append("Возможные целевые столбцы (по названию): " + ", ".join(targets))
    else:
        details_lines.append("Целевые столбцы по названию не определены (это не ошибка).")

    if rec:
        details_lines.append("Возможная цель рекомендателя: " + rec)
    else:
        details_lines.append("Цель рекомендателя по названию не определена (это не ошибка).")

    if targets or rec:
        return CheckResult(
            name="Таргеты должны быть исключены из признаков",
            ok=True,
            details=(
                "\n".join(details_lines) +
                "\nПроверь в коде обучения, что эти столбцы исключаются из X (иначе будет утечка)."
            ),
            severity="INFO",
        )
    return CheckResult(
        name="Таргеты должны быть исключены из признаков",
        ok=True,
        details="\n".join(details_lines),
        severity="INFO",
    )

def find_py_files() -> List[Path]:
    paths = []
    for base in [SRC_DIR, SCRIPTS_DIR, APP_DIR]:
        if base.exists():
            paths.extend(list(base.rglob("*.py")))
    return sorted(set(paths))


def search_patterns_in_code(files: List[Path]) -> List[CheckResult]:
    text_all = {p: _read_text_safe(p) for p in files}

    def has(pattern: str) -> Tuple[bool, List[str]]:
        hits = []
        rx = re.compile(pattern, re.IGNORECASE)
        for p, t in text_all.items():
            if rx.search(t):
                hits.append(str(p.relative_to(ROOT)))
        return (len(hits) > 0, hits)

    results: List[CheckResult] = []

    ok_split, split_files = has(r"test_size\s*=\s*0\.3\b")
    results.append(CheckResult(
        name="В коде указан test_size=0.3 (70/30)",
        ok=ok_split,
        details=("Найдено в:\n" + "\n".join(split_files)) if ok_split else
                "Не найдено 'test_size=0.3'. Убедись, что train/test split = 70/30 по заданию.",
        severity="ERROR" if not ok_split else "INFO",
    ))

    ok_cv, cv_files = has(r"\bcv\s*=\s*10\b")
    results.append(CheckResult(
        name="В коде указан cv=10 (10 блоков CV)",
        ok=ok_cv,
        details=("Найдено в:\n" + "\n".join(cv_files)) if ok_cv else
                "Не найдено 'cv=10'. Убедись, что GridSearchCV использует 10 фолдов по заданию.",
        severity="ERROR" if not ok_cv else "INFO",
    ))

    ok_scaler, scaler_files = has(r"StandardScaler\s*\(")
    results.append(CheckResult(
        name="Используется StandardScaler",
        ok=ok_scaler,
        details=("Найдено в:\n" + "\n".join(scaler_files)) if ok_scaler else
                "Не найден StandardScaler. Если у тебя нормализация обязательна — добавь.",
        severity="WARN" if not ok_scaler else "INFO",
    ))

    ok_pipe, pipe_files = has(r"\bPipeline\s*\(")
    results.append(CheckResult(
        name="Используется Pipeline (желательно для честной CV без утечки)",
        ok=ok_pipe,
        details=("Найдено в:\n" + "\n".join(pipe_files)) if ok_pipe else
                "Pipeline не найден. Это не всегда ошибка, но без Pipeline проще допустить утечку.",
        severity="WARN" if not ok_pipe else "INFO",
    ))

    ok_fit_tr, ft_files = has(r"fit_transform\s*\(")
    if ok_fit_tr:
        results.append(CheckResult(
            name="Найдены вызовы fit_transform (проверь на утечку данных)",
            ok=False,
            details=(
                "fit_transform найден в:\n" + "\n".join(ft_files) +
                "\nЕсли fit_transform применяется ко всему датасету ДО split — это утечка."
            ),
            severity="WARN",
        ))
    else:
        results.append(CheckResult(
            name="Подозрительные fit_transform не найдены",
            ok=True,
            details="Это хорошо: меньше риск утечки через preprocessing.",
            severity="INFO",
        ))

    return results

ARTIFACT_PATTERNS = [
    r".*model.*\.(joblib|pkl|pickle)$",
    r".*scaler.*\.(joblib|pkl|pickle)$",
    r".*features?.*\.(json|txt|joblib|pkl)$",
]


def find_artifacts() -> Tuple[List[Path], List[Path]]:
    found = []
    searched_roots = [MODELS_DIR, ROOT]
    for root in searched_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                for pat in ARTIFACT_PATTERNS:
                    if re.match(pat, p.name, flags=re.IGNORECASE):
                        found.append(p)
                        break
    found = sorted(set(found))
    in_models = [p for p in found if MODELS_DIR in p.parents]
    return found, in_models


def check_artifacts_exist() -> CheckResult:
    found, in_models = find_artifacts()
    if not found:
        return CheckResult(
            name="Артефакты обучения (модель/скейлер/фичи) существуют",
            ok=False,
            details=(
                "Не найдено файлов модели/скейлера/фичей.\n"
                "Запусти обучение: python scripts/train_models_run.py и python scripts/train_recommender_run.py"
            ),
            severity="WARN",
        )

    details = []
    details.append(f"Найдено артефактов всего: {len(found)}")
    if MODELS_DIR.exists():
        details.append(f"В папке models/: {len(in_models)}")
    details.append("Примеры:\n" + "\n".join([str(p.relative_to(ROOT)) for p in found[:10]]))

    if not MODELS_DIR.exists():
        return CheckResult(
            name="Артефакты обучения (модель/скейлер/фичи) существуют",
            ok=True,
            details="\n".join(details) + "\nПапка models/ отсутствует — проверь, куда сохраняются артефакты.",
            severity="WARN",
        )

    return CheckResult(
        name="Артефакты обучения (модель/скейлер/фичи) существуют",
        ok=True,
        details="\n".join(details),
        severity="INFO",
    )


def check_import_src_package() -> CheckResult:

    init_file = SRC_DIR / "__init__.py"
    if not init_file.exists():
        return CheckResult(
            name="src является пакетом (наличие src/__init__.py)",
            ok=False,
            details=(
                "Не найден src/__init__.py.\n"
                "Без него часто ломаются импорты 'from src...'. Создай пустой файл src/__init__.py"
            ),
            severity="WARN",
        )
    return CheckResult(
        name="src является пакетом (наличие src/__init__.py)",
        ok=True,
        details="src/__init__.py найден — импорты будут стабильнее.",
        severity="INFO",
    )


def check_streamlit_entry() -> CheckResult:
    app = APP_DIR / "streamlit_app.py"
    if not app.exists():
        return CheckResult(
            name="Наличие app/streamlit_app.py",
            ok=False,
            details="Файл app/streamlit_app.py не найден.",
            severity="WARN",
        )
    txt = _read_text_safe(app)

    return CheckResult(
        name="Точка входа Streamlit",
        ok=True,
        details=(
            "Streamlit запускается командой:\n"
            "  streamlit run app/streamlit_app.py\n"
            "Если запускать через python, будет предупреждение ScriptRunContext — это нормально."
        ),
        severity="INFO",
    )

def main() -> int:
    print("=== SELF CHECK: дипломный проект композиты ===")
    print(f"Project root: {ROOT}")
    print()

    results: List[CheckResult] = []

    r = check_data_files_exist()
    results.append(r)
    if not r.ok:
        for x in results:
            _print_result(x)
        return 1

    try:
        df = load_and_merge_inner_by_index(DATA_BP, DATA_NUP)
    except Exception as e:
        results.append(CheckResult(
            name="Загрузка и объединение Excel",
            ok=False,
            details=f"Ошибка чтения/объединения: {e}",
            severity="ERROR",
        ))
        for x in results:
            _print_result(x)
        return 1

    results.append(check_merge_inner_index(df))
    results.append(check_missing_values(df))
    results.append(check_targets_not_in_features(df))

    py_files = find_py_files()
    if not py_files:
        results.append(CheckResult(
            name="Поиск python-файлов",
            ok=False,
            details="Не найдены .py файлы в src/scripts/app — проверь структуру проекта.",
            severity="ERROR",
        ))
    else:
        results.append(CheckResult(
            name="Поиск python-файлов",
            ok=True,
            details=f"Найдено .py файлов: {len(py_files)}",
            severity="INFO",
        ))
        results.extend(search_patterns_in_code(py_files))

    results.append(check_artifacts_exist())

    results.append(check_import_src_package())
    results.append(check_streamlit_entry())

    # Print all
    print("=== РЕЗУЛЬТАТЫ ПРОВЕРКИ ===")
    for x in results:
        _print_result(x)

    errors = [x for x in results if (not x.ok and x.severity == "ERROR")]
    warns = [x for x in results if (not x.ok and x.severity == "WARN")]

    print()
    print("=== ИТОГ ===")
    if errors:
        print(f"❌ Ошибок: {len(errors)}. Проект нужно поправить по пунктам выше.")
        return 1
    if warns:
        print(f"⚠️ Предупреждений: {len(warns)}. Проект работает, но есть риски/рекомендации.")
        return 0
    print("✅ Всё выглядит корректно: критичных проблем не найдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())