# Система мониторинга академических рисков студентов

Интерактивная платформа для **сбора, обработки, анализа и визуализации** данных о студентах (успеваемость, анкеты, психометрика, опросники) с применением **классических и ансамблевых моделей машинного обучения**, **объяснимости (SHAP)** и **мониторинга дрейфа данных**.

Основной пользовательский интерфейс — **`app.py` (Streamlit)**. Архитектура поддерживает как standalone-режим, так и микросервисное развёртывание через **FastAPI + Redis RQ + Docker**.

---

## Содержание

1. [Возможности](#возможности)
2. [Архитектура проекта](#архитектура-проекта)
3. [Структура репозитория](#структура-репозитория)
4. [Пакет `ml_core`](#пакет-ml_core)
5. [API и фоновые задачи](#api-и-фоновые-задачи)
6. [Мониторинг и конфигурация](#мониторинг-и-конфигурация)
7. [Установка и запуск](#установка-и-запуск)
8. [Данные: форматы и ожидания](#данные-форматы-и-ожидания)
9. [Артефакты и логи](#артефакты-и-логи)

---

## Возможности

### Загрузка данных
- **Синтетические наборы** по 7 категориям: успеваемость (`grades`), психология (`psychology`), креативность (`creativity`), ценности (`values`), личность (`personality`), активность (`activities`), карьера (`career`)
- **CSV** с произвольными признаками и целевой переменной (`risk_flag` / `target`)
- **Excel** с многлистовыми опросниками (Вильямс, Шварц, социодемография и др.) через автоматическое определение типа листа и объединение по `user_id`

### Предобработка и feature engineering
- **Композитные признаки**: `trend_grades`, `grade_stability`, `cognitive_load`, `overall_satisfaction`, `psychological_wellbeing`, `academic_activity`
- **Комбинации признаков**: автоматическое создание sum/diff/ratio/product для числовых, конкатенация для текстовых
- **Композитный скоринг**: взвешенная нормализованная сумма признаков с квантильным рейтингом (1–100)
- **SMOTE** с адаптивным `k_neighbors`, автоматическое отключение при малых классах
- **Обработка пропусков**: стратегии `auto`, `mean`, `median`, `mode`, `interpolate`, `drop_rows`, `drop_columns`
- **Детекция выбросов**: IQR и Z-score методы
- **Текстовые признаки**: длина, количество слов, сложность текста

### Аналитика
- **Корреляционный анализ**: расширенная матрица с отбором сильных корреляций (≥ порога), топ корреляций с target, Plotly heatmap
- **K-means кластеризация**: профили кластеров (средние значения, размеры, проценты), PCA-визуализация (2D через Plotly и Matplotlib)
- **Кросс-таблицы**: сводные таблицы, χ²-тест, heatmap, stacked bar chart, экспорт в CSV/Excel
- **Временные ряды**: траектории студентов (линейный тренд → improving/declining/stable), анализ когорт, детекция негативной динамики, прогноз оценок (линейная экстраполяция)

### Модели машинного обучения
- **Logistic Regression**, **Random Forest**, **XGBoost**
- **Кросс-валидация** (StratifiedKFold, 5 folds) по F1-score
- **RandomizedSearchCV** для XGBoost (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)
- **Отбор признаков**: комбинация ANOVA F-value + Random Forest importance; mutual_info_classif + RFE
- **Метрики**: F1, Precision, Recall, ROC-AUC
- **Визуализация**: ROC-кривые, Confusion Matrix, Feature Importance (Plotly)

### Объяснимость (SHAP)
- **TreeExplainer** для XGBoost/RandomForest, **LinearExplainer** для линейных моделей
- Summary plot, текстовые объяснения для топ-N студентов
- Детальные объяснения с рекомендациями для конкретного наблюдения
- Batch-объяснения для студентов с наибольшим риском

### Мониторинг
- **Дрейф данных**: KS-тест для числовых признаков, χ² для категориальных; отчёт с процентом дрейфа, списком признаков, рекомендациями, оценкой качества данных
- **Фоновый мониторинг**: `DriftMonitorThread` (периодическая проверка), `DriftMonitorScheduler` (несколько моделей)
- **Логирование**: события ML (`ml_events.log`), история метрик моделей (`model_metrics.csv`)

### LLM (опционально)
- **YandexGPT** (реализовано) и **GigaChat** (заглушка)
- Интерпретация кластеров, анализ текстовых ответов, генерация отчётов, ответы на вопросы по данным

---

## Архитектура проекта

Проект поддерживает **два режима работы**:

### Режим 1: Standalone (Streamlit)
```
app.py → ml_core → файлы/модели/логи
```
Всё в одном процессе. Подходит для локального анализа и прототипирования.

### Режим 2: Микросервисный (Docker)
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│  FastAPI API │────▶│  RQ Worker   │
│  (React/Vue) │     │  (uvicorn)   │     │  (Redis RQ)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐     ┌──────────────┐
                     │    Redis     │     │  PostgreSQL  │
                     │  (cache/RQ)  │     │   (БД)       │
                     └──────────────┘     └──────────────┘
```
- **API** (`api/`) — FastAPI с эндпоинтами анализа
- **Worker** (`workers/`) — фоновые задачи через Redis RQ (обучение моделей, SHAP)
- **3 Dockerfile**: `Dockerfile.api`, `Dockerfile.frontend`, `Dockerfile.worker`

---

## Структура репозитория

| Путь | Назначение |
|------|------------|
| **`app.py`** | Streamlit-приложение: сайдбар, пайплайн анализа, визуализация, мониторинг дрейфа, экспорт |
| **`ml_core/`** | Ядро ML-системы (22 модуля, см. ниже) |
| **`api/`** | FastAPI бэкенд: роутеры, Pydantic-схемы |
| **`workers/`** | Фоновые задачи RQ (обучение, SHAP) |
| **`config/`** | Настройки проекта: пути, константы, URL БД/Redis |
| **`monitoring/`** | Пакет-заглушка (мониторинг перенесён в `ml_core/drift_detector.py`) |
| **`prompts/`** | Шаблоны промптов для LLM (кластеры, отчёты, анализ текста) |
| **`models/`** | Сохранённые модели (`.pkl`) и метаданные (`.json`) |
| **`logs/`** | `ml_events.log`, `model_metrics.csv`, `drift_reports/*.json` |
| **`experiments/`** | Сохранённые эксперименты (`metadata.json`, `model.pkl`, `predictions.csv`) |
| **`analysis_data/`** | Синтетика и мониторинговые выгрузки |

---

## Пакет `ml_core`

### Загрузка и подготовка данных

| Модуль | Назначение |
|--------|------------|
| **`data.py`** | Генерация синтетических данных (7 категорий), загрузка CSV, подготовка train/test split, генерация temporal-данных с динамикой по семестрам |
| **`loader.py`** | Парсинг многлистовых Excel: автоопределение типа листа (Вильямс, Шварц, демография и др.), объединение по `user_id` |
| **`features.py`** | Композитные признаки, отбор признаков (F-value + RF, MI + RFE), SMOTE, комбинации признаков, композитный скоринг |
| **`imputation.py`** | Обработка пропусков (5 стратегий + auto), детекция выбросов (IQR, Z-score) |
| **`text_processor.py`** | Извлечение признаков из текстовых столбцов (длина, слова, сложность) |

### Модели и обучение

| Модуль | Назначение |
|--------|------------|
| **`models.py`** | `ModelTrainer` — LR, RF, XGBoost; CV, обучение, тюнинг XGBoost (RandomizedSearchCV), сохранение/загрузка |
| **`evaluation.py`** | Метрики (F1, ROC-AUC, Precision, Recall), ROC-кривые, Confusion Matrix, Feature Importance, **SHAP-объяснения** (TreeExplainer/LinearExplainer) |
| **`analyzer.py`** | `ResearchAnalyzer` — единая точка входа, orchestrator полного пайплайна: данные → признаки → кластеры → модели → SHAP → графики |

### Аналитика

| Модуль | Назначение |
|--------|------------|
| **`analysis.py`** | K-means кластеризация, профили кластеров, PCA-визуализация, корреляционный анализ (enhanced версия с отбором сильных корреляций) |
| **`crosstab.py`** | Кросс-таблицы, χ²-тест, heatmap, stacked bar, мульти-кросстаб |
| **`timeseries.py`** | Траектории студентов, анализ когорт, детекция негативной динамики, прогноз оценок, временные признаки (лаги, изменения) |

### Мониторинг и инфраструктура

| Модуль | Назначение |
|--------|------------|
| **`drift_detector.py`** | `DataDriftDetector` — KS-тест (числовые), χ² (категориальные), отчёт с рекомендациями; `DriftMonitorThread`, `DriftMonitorScheduler` |
| **`logger.py`** | `MLLogger` — логирование событий (JSON), метрик моделей (CSV), предсказаний |
| **`error_handler.py`** | Глобальный логгер `ml_core`, `safe_execute()` для безопасного вызова функций |
| **`experiment_tracker.py`** | `ExperimentTracker` — сохранение/загрузка/удаление экспериментов, каталог `experiments/<id>/` |

### Конфигурация и схемы

| Модуль | Назначение |
|--------|------------|
| **`config.py`** | `Config` — пути (`DATA_DIR`, `MODELS_DIR`, `LOGS_DIR`, `EXPERIMENTS_DIR`), константы (`RANDOM_SEED`, `CV_FOLDS`, `TEST_SIZE`, `SHAP_TOP_N`) |
| **`schemas.py`** | Pydantic-модели: `AnalysisRequest`, `AnalysisResult`, `CompositeScoreRequest`, `TrajectoryRequest` |
| **`llm_interface.py`** | `LLMInterface` — YandexGPT (реализовано), GigaChat (заглушка); интерпретация кластеров, отчёты, Q&A |

### Вспомогательные модули

| Модуль | Назначение |
|--------|------------|
| **`composite_score.py`** | Заготовка (пустой файл) |
| **`feature_combinations.py`** | Заготовка (пустой файл, логика перенесена в `features.py`) |

---

## API и фоновые задачи

### FastAPI (`api/`)

Роутер: `prefix="/api/analyze"`

| Эндпоинт | Метод | Запрос | Ответ | Описание |
|----------|-------|--------|-------|----------|
| `/api/analyze/full` | POST | `AnalysisRequest` (JSON с массивом записей) | `AnalysisResponse` | Полный пайплайн: признаки → кластеры → модели → SHAP → графики |
| `/api/analyze/composite/create` | POST | `CompositeRequest` (df + веса признаков) | `{score_name, statistics}` | Создание композитного скоринга |
| `/api/analyze/subset/select` | POST | `SubsetRequest` (df + условие/семплирование) | `{count, data}` | Выборка подмножества данных |

> **Примечание:** `api/main.py` закомментирован. Для запуска API необходимо раскомментировать `main.py` и подключить роутер через `app.include_router()`.

### Фоновые задачи RQ (`workers/tasks.py`)

| Задача | Вход | Прогресс | Результат |
|--------|------|----------|-----------|
| `train_model_task(data_path, params)` | Путь к CSV | loading (10%) → preprocessing (30%) → training (60%) → saving (90%) | `model_id`, `model_path`, `metrics`, `features` |
| `shap_task(model_id, data_path, threshold)` | ID модели + путь к CSV | loading_model (20%) → loading_data (40%) → computing_shap (70%) | Список SHAP-объяснений |

Результаты кэшируются в Redis с ключом `task_result:{job_id}` (TTL 24 часа).

---

## Мониторинг и конфигурация

### `config/settings.py`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `BASE_DIR` | Корень проекта | Загружается из `.env` через `python-dotenv` |
| `DEBUG` | `False` | Режим отладки |
| `DATABASE_URL` | `sqlite:///db.sqlite3` | URL базы данных |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis для кэша/RQ |
| `RANDOM_SEED` | `42` |_seed для воспроизводимости |
| `DEFAULT_N_CLUSTERS` | `3` | Число кластеров по умолчанию |
| `RISK_THRESHOLD` | `0.5` | Порог классификации риска |

### Переменные окружения (`.env`)

| Переменная | Назначение |
|------------|------------|
| `DEBUG` | `true`/`false` |
| `DATABASE_URL` | URL БД для бэкенда |
| `REDIS_URL` | Redis для кэша/очередей |
| `LLM_API_KEY` | Ключ для YandexGPT |
| `YANDEX_FOLDER_ID` | Каталог Yandex Cloud для YandexGPT |

---

## Установка и запуск

### Требования
- Python **3.10+** (рекомендуется 3.11)

### Standalone (Streamlit)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

Откройте `http://localhost:8501`.

### Микросервисное развёртывание (Docker)

> Требует `docker-compose.yml` (если добавлен в проект) или ручного запуска контейнеров:

```bash
docker build -t ml-api -f Dockerfile.api .
docker build -t ml-worker -f Dockerfile.worker .
docker build -t ml-frontend -f Dockerfile.frontend .
```

Необходимые сервисы: **Redis** (кэш/RQ), **PostgreSQL/SQLite** (БД).

---

## Данные: форматы и ожидания

### CSV
- Числовые и категориальные признаки
- Целевая переменная: **`risk_flag`** (предпочтительно) или **`target`** (бинарная)
- Для временных рядов: **`student_id`**, **`semester`**

### Excel (опросники)
- Листы с ключевыми словами: `williams`, `schwartz`, `demographics`, `personality`, `grades`, `career` и др.
- Колонка респондента: `user`, `user_id` или `VK_id` (иначе создаётся искусственный `user_id`)
- Шкалы Лайкерта автоматически приводятся к числовому типу

### Синтетические данные
- **7 категорий**: grades, psychology, creativity, values, personality, activities, career
- **Агрегированный набор** (`_generate_synthetic_data_tz`): 1200 студентов со всеми признаками (демография + успеваемость + анкеты + психология + эссе + креативность + ценности + личность + активность + `risk_flag`)
- **Генерация дрейфа**: `generate_two_sets=True` создаёт reference и new данные со смещением распределений

---

## Артефакты и логи

| Путь | Содержимое |
|------|------------|
| `models/*.pkl` | Сохранённые модели (`joblib`) |
| `models/*_meta.json` | Метаданные обучения (метрики, признаки, время) |
| `logs/ml_events.log` | События MLLogger (JSON-строки) |
| `logs/model_metrics.csv` | История метрик по запускам (timestamp, model_name, f1, roc_auc, precision, recall) |
| `logs/drift_reports/*.json` | Отчёты дрейфа (drift_percentage, drifted_features, recommendations) |
| `experiments/<id>/` | Эксперименты: `metadata.json`, `model.pkl`, `explanations.csv`, `predictions.csv` |
| `analysis_data/synthetic/` | Выгрузки синтетических данных |
| `analysis_data/monitoring/` | Мониторинговые выгрузки |
| `temp_excel.xlsx` | Временный файл при загрузке Excel из Streamlit |

---

## Краткая карта соответствия ТЗ (ML-часть)

| Требование | Реализация |
|------------|-----------|
| Представление данных для анализа | Агрегация и признаки (`data.py`, `features.py`), заполнение пропусков (`imputation.py`), подготовка выборок |
| Аналитическая система | Корреляция, кластеризация, классификация, отбор признаков, визуализация в Streamlit |
| Интерпретируемость | SHAP-объяснения (TreeExplainer/LinearExplainer), текстовые объяснения с рекомендациями |
| Надёжность | Логи метрик (`MLLogger`), мониторинг дрейфа (`DataDriftDetector`), кросс-валидация |
| АРМ исследователя | `ResearchAnalyzer` — единая точка входа с полным пайплайном |
| Эксперименты | `ExperimentTracker` — сохранение, загрузка, каталог экспериментов |
| Прогнозирование | `forecast_grades` в `timeseries.py`, траектории студентов |
