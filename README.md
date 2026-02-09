# Диплом: Прогнозирование конечных свойств композитов

## Запуск
1) Открой папку проекта в PyCharm
2) Создай venv
3) Установи зависимости:
```bash
pip install -r requirements.txt
```

## Скрипты (запуск через Run)
- scripts/eda.py — EDA: гистограммы, boxplot, pairplot, корреляции
- scripts/train_models_run.py — обучение моделей (GridSearchCV, KFold=10, train/test=70/30), сохранение best_model.pkl
- scripts/train_recommender_run.py — обучение нейросети-рекомендателя, сохранение recommender_model.pkl и recommender_scaler.pkl
- app/streamlit_app.py — Streamlit UI (запуск из терминала):
```bash
streamlit run app/streamlit_app.py
```
