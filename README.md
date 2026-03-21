# Система мониторинга академических рисков студентов

Интерактивная платформа для анализа и прогнозирования академических рисков студентов на основе методов машинного обучения.

## Возможности

- 📊 **Корреляционный анализ** признаков
- 🎯 **Кластеризация студентов** на группы
- 🤖 **Сравнение моделей ML** (Логистическая регрессия, Random Forest, XGBoost)
- ⚙️ **Оптимизация гиперпараметров** XGBoost
- 📈 **Визуализация метрик** (ROC-кривые, Confusion Matrix, Feature Importance)
- 💡 **Объяснения предсказаний** на основе SHAP
- 💾 **Экспорт результатов** в CSV

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск

### Командная строка (скрипт)
```bash
python student_risk_pipeline.py
```

### Веб-интерфейс (React, основной)
```bash
cd frontend
npm install
npm run dev
```

Приложение откроется в браузере по адресу `http://localhost:5173`

### Legacy режим (Streamlit, опционально)
```bash
streamlit run app.py
```

## Структура проекта

- `student_risk_pipeline.py` - основной модуль с функциями анализа
- `app.py` - Streamlit legacy интерфейс (для отладки/сравнения)
- `api/main.py` - FastAPI backend для React/внешних клиентов
- `workers/tasks.py` - фоновые задачи RQ (обучение, SHAP)
- `frontend/` - React (Vite) frontend (основной UI)
- `requirements.txt` - зависимости проекта

## Использование

### Через React-интерфейс (основной сценарий):

1. Запустите backend (`uvicorn api.main:app ...`) и worker (`rq worker ...`)
2. Запустите `frontend` (`npm run dev`)
3. Загрузите CSV
4. Запустите:
   - корреляционный анализ,
   - обучение модели (фоновой task),
   - SHAP объяснения (фоновая task)
5. Отслеживайте статус и результат задач в UI

### Формат данных CSV:

Файл должен содержать:
- Числовые признаки для анализа
- Целевую переменную `risk_flag` (обязательно для обучения)

## Результаты

После выполнения анализа создаются:
- `corr_matrix.csv` - корреляционная матрица
- `corr_heatmap.png` - визуализация корреляций
- `cluster_profiles.csv` - профили кластеров
- `clusters_2d.png` - визуализация кластеров
- `roc_curves.png` - ROC-кривые моделей
- `cm_best.png` - confusion matrix лучшей модели
- `predictions.csv` - предсказания модели
- `detailed_explanations.csv` - детальные объяснения

## Технологии

- Python 3.8+
- Streamlit
- FastAPI
- Redis + RQ
- React (Vite)
- scikit-learn
- XGBoost
- SHAP
- Plotly
- Pandas, NumPy

## Миграция на React + FastAPI

### Что уже готово
- FastAPI endpoints:
  - `POST /api/ml/train` - запуск обучения в фоне
  - `GET /api/ml/tasks/{task_id}` - статус фоновой задачи
  - `POST /api/ml/shap` - запуск SHAP в фоне
  - `POST /api/ml/correlation` - синхронная корреляция для небольших CSV
- Redis + RQ:
  - `workers/tasks.py` выполняет обучение и генерацию объяснений в фоне
- React frontend (минимальный рабочий):
  - `frontend/src/App.jsx` — загрузка CSV, запуск задач, polling статусов

### Локальный запуск новой архитектуры
1. Установить Python зависимости:
```bash
pip install -r requirements.txt
```
2. Поднять Redis:
```bash
docker run --rm -p 6379:6379 redis:7-alpine
```
3. Запустить API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```
4. Запустить воркер:
```bash
rq worker --url redis://localhost:6379/0
```
5. Запустить React:
```bash
cd frontend
npm install
npm run dev
```

### Docker-вариант
```bash
docker compose up --build
```

