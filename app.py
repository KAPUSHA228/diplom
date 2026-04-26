"""
⚠️ DEPRECATED — Legacy Streamlit UI

Данное приложение устарело. Основной интерфейс — React MFE (frontend/).
Файл сохранён для справки и локального анализа.
Основной режим запуска: React + FastAPI (см. README.md).

Дата перевода на React: Апрель 2026.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import re
from ml_core.analyzer import ResearchAnalyzer

# Импортируем ВАШИ модули
from ml_core.data import load_from_csv
from ml_core.features import add_composite_features
from ml_core.analysis import cluster_students, plot_clusters_pca
from ml_core.models import ModelTrainer
from ml_core.evaluation import plot_roc_curves, plot_confusion_matrix, plot_feature_importance
from ml_core.logger import MLLogger
from ml_core.drift_detector import DataDriftDetector
from ml_core.analysis import correlation_analysis
from ml_core.features import build_composite_score
from ml_core.timeseries import forecast_grades
from ml_core.crosstab import export_crosstab
from ml_core.error_handler import safe_execute, logger
from ml_core.utils import save_plotly_fig

# Инициализация session_state
if "results_saved" not in st.session_state:
    st.session_state.results_saved = False
    st.session_state.analysis_completed = False
    st.session_state.df = None
    st.session_state.all_features = None
    st.session_state.cluster_labels = None
    st.session_state.cluster_profiles = None
    st.session_state.fig_clusters = None
    st.session_state.fig_corr = None
    st.session_state.X_train_sel = None
    st.session_state.X_test_sel = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.selected_cols = None
    st.session_state.cv_df = None
    st.session_state.best_model = None
    st.session_state.best_name = None
    st.session_state.metrics = None
    st.session_state.test_metrics = None
    st.session_state.models_for_plot = None
    st.session_state.fig_roc = None
    st.session_state.y_pred = None
    st.session_state.fig_cm = None
    st.session_state.fig_fi = None
    st.session_state.explanations = None
    st.session_state.target_column = "risk_flag"
    st.session_state.data_category = "grades"
    st.session_state.target_selected = False  # Флаг выбора целевой переменной
    st.session_state.data_loaded = False  # Флаг загрузки данных
    st.session_state.raw_df = None  # Исходные данные без обработки

    if "drift_detector" not in st.session_state:
        st.session_state.drift_detector = None
        st.session_state.drift_report = None
        st.session_state.reference_data = None
        if "monitor_thread" not in st.session_state:
            st.session_state.monitor_thread = None

# Инициализация логгера
ml_logger = MLLogger()

# Настройка страницы
st.set_page_config(page_title="Система мониторинга академических рисков", page_icon="📊", layout="wide")

st.title("📊 Система мониторинга академических рисков студентов")
st.markdown("---")

# Инициализация трейнера моделей
trainer = ModelTrainer(models_dir="models")
if "analyzer" not in st.session_state:
    st.session_state.analyzer = ResearchAnalyzer()
analyzer = st.session_state.analyzer
# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")

    # Загрузка данных
    st.subheader("📁 Загрузка данных")

    data_type = st.radio("Тип данных", ["Синтетические данные", "CSV файл", "Excel файл"], key="data_type_radio")

    df = None
    uploaded_file = None
    excel_file = None

    if data_type == "Синтетические данные":
        st.session_state.synthetic_category = st.selectbox(
            "Категория данных",
            ["grades", "psychology", "creativity", "values", "personality", "activities", "career"],
            format_func=lambda x: {
                "grades": "📚 Успеваемость (риск отчисления)",
                "psychology": "🧠 Психологическое состояние (риск выгорания)",
                "creativity": "🎨 Креативность (высокая креативность)",
                "values": "💎 Ценности Шварца (тип профиля)",
                "personality": "👤 Личностные опросы (потенциал лидерства)",
                "activities": "⚡ Активность (активный участник)",
                "career": "💼 Карьерные намерения (определённость)",
            }.get(x, x),
        )
        st.session_state.synthetic_n_students = st.slider("Количество студентов", 100, 2000, 500)
        st.session_state.generate_drift_data = st.checkbox("Сгенерировать данные для дрейфа", value=True)

        # Автоматически загружаем синтетические данные при изменении параметров
        if st.button("🔄 Сгенерировать данные", key="gen_synth"):
            with st.spinner("Генерация синтетических данных..."):
                from ml_core.data import load_data

                category = st.session_state.synthetic_category
                n_students = st.session_state.synthetic_n_students
                generate_drift_data = st.session_state.generate_drift_data

                result = load_data(category=category, n_students=n_students, generate_two_sets=generate_drift_data)

                st.session_state.raw_df = result["data"]
                st.session_state.data_category = category
                st.session_state.target_column = result["target"]
                st.session_state.data_loaded = True
                st.session_state.target_selected = True  # Для синтетики цель уже выбрана
                st.session_state.analysis_completed = False  # Сбрасываем анализ

                if generate_drift_data and "reference" in result and "new" in result:
                    st.session_state.drift_reference = result["reference"]
                    st.session_state.drift_new_data = result["new"]

                st.success(f"✅ Синтетические данные загружены ({len(result['data'])} записей)")
                st.rerun()

    elif data_type == "CSV файл":
        uploaded_file = st.file_uploader(
            "Выберите CSV файл",
            type=["csv"],
            help="Файл должен содержать колонки с признаками и целевую переменную 'risk_flag'",
            key="csv_uploader",
        )
        if uploaded_file is not None:
            df = load_from_csv(uploaded_file)
            st.session_state.raw_df = df
            st.session_state.data_loaded = True
            st.session_state.target_selected = False  # Нужно выбрать цель
            st.session_state.analysis_completed = False
            st.success(f"✅ CSV файл загружен ({len(df)} записей)")

    elif data_type == "Excel файл":
        excel_file = st.file_uploader(
            "Выберите Excel файл", type=["xlsx", "xls"], help="Файл с опросом/опросами", key="excel_uploader"
        )
        if excel_file is not None:
            temp_path = "temp_excel.xlsx"
            # Сохраняем временно
            with open(temp_path, "wb") as f:
                f.write(excel_file.getbuffer())

            merge_all = st.checkbox(
                "Объединить все листы по user / user_id / *_id",
                value=False,
                help="Включает режим полного мёрджa всех листов из файла",
            )
            if merge_all:
                if st.button("📂 Объединить все листы", type="primary"):
                    with st.spinner("Выполняется объединение всех листов..."):
                        from ml_core.loader import preprocess_excel_data

                        df_merged, msg = preprocess_excel_data(temp_path)

                        if df_merged is not None:
                            st.session_state.raw_df = df_merged
                            st.session_state.data_loaded = True
                            st.session_state.target_selected = False
                            st.session_state.analysis_completed = False
                            st.session_state.data_category = "excel_merged"
                            st.success(f"✅ {msg}. Объединено {len(df_merged)} записей.")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
            else:
                from ml_core.loader import get_sheet_names, load_excel_sheet

                sheet_names = get_sheet_names(temp_path)

                if "selected_sheet" not in st.session_state:
                    st.session_state.selected_sheet = sheet_names[0] if sheet_names else None

                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox(
                        "Выберите лист для анализа",
                        sheet_names,
                        index=(
                            sheet_names.index(st.session_state.selected_sheet)
                            if st.session_state.selected_sheet in sheet_names
                            else 0
                        ),
                        key="sheet_selector",
                    )
                    st.session_state.selected_sheet = selected_sheet
                else:
                    selected_sheet = sheet_names[0]

                if st.button("📂 Загрузить выбранный лист", key="load_sheet"):
                    df, msg = load_excel_sheet("temp_excel.xlsx", selected_sheet)
                    if df is not None:
                        st.session_state.raw_df = df
                        st.session_state.data_loaded = True
                        st.session_state.target_selected = False
                        st.session_state.analysis_completed = False
                        st.session_state.data_category = "excel"
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    st.markdown("---")

    # Настройки анализа (показываем только если данные загружены)
    if st.session_state.data_loaded:
        st.subheader("🔧 Параметры анализа")
        n_clusters = st.slider("Количество кластеров", 2, 6, 3)
        risk_threshold = st.slider("Порог риска", 0.0, 1.0, 0.5, 0.05)
        use_hp_tuning = st.checkbox("Использовать оптимизацию гиперпараметров", value=False)
        n_iter_tuning = st.slider("Итераций для оптимизации", 10, 50, 20) if use_hp_tuning else 20

        st.markdown("---")

        st.subheader("🎛️ Дополнительные настройки")

        # Выбор метрики для оптимизации
        optimization_metric = st.selectbox("Метрика для оптимизации", ["f1", "roc_auc", "precision", "recall"], index=0)
        use_smote = st.checkbox("Использовать SMOTE для балансировки классов", value=True)

        corr_threshold = st.slider("Порог корреляции для сильных связей", 0.0, 1.0, 0.3, 0.05)
        # Количество признаков для отбора
        n_features_to_select = st.slider("Количество признаков в финальной модели", min_value=3, max_value=15, value=7)

        # Выбор моделей для сравнения
        st.write("**Модели для обучения:**")
        use_lr = st.checkbox("Logistic Regression", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_xgb = st.checkbox("XGBoost", value=True)

        # Порог для SHAP-объяснений
        shap_top_n = st.slider("Количество факторов в объяснениях", min_value=3, max_value=10, value=5)

        # Кнопка запуска
        run_analysis = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

# ==================== ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ====================
# Отображаем выбор целевой переменной, если данные загружены и цель не выбрана
if st.session_state.data_loaded and not st.session_state.target_selected:
    st.subheader("🎯 Выбор целевой переменной")

    df_raw = st.session_state.raw_df

    # Исключаем служебные колонки
    exclude = ["user", "user_id", "VK_id", "дата", "date", "_source_sheet", "_sheet_type"]
    candidate_cols = [col for col in df_raw.columns if col not in exclude]

    if not candidate_cols:
        st.error("Нет подходящих колонок для целевой переменной")
        st.stop()

    # Показываем информацию о данных
    st.write(f"**Всего записей:** {len(df_raw)}")
    st.write(
        f"**Колонки в данных:** {', '.join(df_raw.columns.tolist()[:10])}{'...' if len(df_raw.columns) > 10 else ''}"
    )

    # Выбор целевой переменной
    target_choice = st.selectbox(
        "Колонка для предсказания (целевая переменная)",
        candidate_cols,
        help="Можно выбрать любую колонку – система построит модель предсказания для неё.",
    )

    # Показываем статистику по выбранной колонке
    if target_choice in df_raw.columns:
        col_data = df_raw[target_choice]
        st.write("**Статистика выбранной колонки:**")
        st.write(f"- Тип данных: {col_data.dtype}")
        st.write(f"- Уникальных значений: {col_data.nunique()}")

        if col_data.dtype in ["int64", "float64"]:
            st.write(f"- Диапазон: {col_data.min():.2f} - {col_data.max():.2f}")
            # Предлагаем преобразовать в бинарную переменную
            if col_data.nunique() > 2:
                st.info(
                    f"Колонка содержит {col_data.nunique()} уникальных значений. Будет преобразована в бинарную (порог = медиана)"
                )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Подтвердить выбор цели", type="primary", use_container_width=True):
            st.session_state.target_column = target_choice
            st.session_state.target_selected = True
            st.session_state.data_category = "custom"
            st.rerun()
    with col2:
        if st.button("🔄 Сбросить данные", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.target_selected = False
            st.session_state.raw_df = None
            st.session_state.analysis_completed = False
            st.rerun()

    st.stop()  # Останавливаем выполнение, пока цель не выбрана

# ==================== ВЫЧИСЛЕНИЯ (только при нажатии кнопки) ====================
if st.session_state.data_loaded and st.session_state.target_selected:
    # Проверяем, нужно ли запустить анализ
    run_analysis_flag = (
        st.sidebar.button("🚀 Запустить анализ", type="primary", use_container_width=True)
        if "run_analysis" not in locals()
        else run_analysis
    )

    if run_analysis_flag or not st.session_state.analysis_completed:
        # Логируем запуск
        ml_logger.log_event(
            "analysis_started",
            {"data_type": data_type, "n_clusters": n_clusters, "use_hp_tuning": use_hp_tuning, "use_smote": use_smote},
        )

        with st.spinner("Выполняется полный анализ через ResearchAnalyzer..."):
            try:
                # Определяем флаги
                is_synthetic = data_type == "Синтетические данные"

                # Создаём анализатор
                analyzer = ResearchAnalyzer()

                # Запускаем полный анализ через главный фасад
                analysis_result = analyzer.run_full_analysis(
                    df=st.session_state.raw_df.copy(),
                    target_col=st.session_state.target_column,
                    n_clusters=n_clusters,
                    risk_threshold=risk_threshold,
                    corr_threshold=corr_threshold,
                    is_synthetic=is_synthetic,
                    use_smote=use_smote,
                )

                if analysis_result.status == "error":
                    st.error(f"❌ Ошибка при выполнении анализа: {analysis_result.message}")
                    st.stop()

                # Извлекаем основные результаты
                metrics = analysis_result.metrics
                test_metrics = analysis_result.test_metrics or metrics.get("test", {})
                all_features = analysis_result.selected_features
                explanations = analysis_result.explanations or []

                # Сохраняем cv_results для отображения
                if hasattr(analysis_result, "cv_results"):
                    st.session_state.cv_results = analysis_result.cv_results
                else:
                    st.session_state.cv_results = metrics.get("cv_results", {})

                # Для отображения
                df = st.session_state.raw_df.copy()
                df = safe_execute(add_composite_features, df)

                cluster_labels, _, _ = safe_execute(
                    cluster_students, df, n_clusters=n_clusters, feature_cols=all_features
                )
                df["cluster"] = cluster_labels

                fig_clusters = plot_clusters_pca(df, cluster_labels, all_features)
                cluster_profiles = pd.DataFrame(analysis_result.cluster_profiles)

                # === ГРАФИКИ ===
                fig_cm = (
                    plot_confusion_matrix(
                        analyzer.last_y_test, analyzer.last_y_pred, analyzer.last_model_name or "Best Model"
                    )
                    if hasattr(analyzer, "last_y_test")
                    else None
                )

                fig_roc = None
                if hasattr(analyzer, "last_X_test") and hasattr(analyzer, "last_y_test"):
                    model_for_roc = analyzer.trainer.models.get(analyzer.last_model_name)
                    if model_for_roc is not None:
                        fig_roc = plot_roc_curves(
                            {analyzer.last_model_name or "Best": model_for_roc},
                            analyzer.last_X_test,
                            analyzer.last_y_test,
                        )

                fig_fi = (
                    plot_feature_importance(analyzer.trainer.models.get(analyzer.last_model_name), all_features)
                    if hasattr(analyzer, "last_model_name") and analyzer.last_model_name
                    else None
                )

                # ==================== СОХРАНЕНИЕ ====================
                st.session_state.analysis_completed = True
                st.session_state.df = df
                st.session_state.all_features = all_features
                st.session_state.cluster_labels = cluster_labels
                st.session_state.cluster_profiles = cluster_profiles
                st.session_state.fig_clusters = fig_clusters
                st.session_state.test_metrics = test_metrics
                st.session_state.explanations = explanations
                st.session_state.best_name = analyzer.last_model_name or "Best Model"
                st.session_state.best_model = analyzer.trainer.models.get(analyzer.last_model_name)
                st.session_state.fig_cm = fig_cm
                st.session_state.fig_roc = fig_roc
                st.session_state.fig_fi = fig_fi
                st.session_state.X_test_sel = analyzer.last_X_test
                st.session_state.y_pred = analyzer.last_y_pred
                st.session_state.selected_cols = all_features

                # Drift Detector
                if st.session_state.drift_detector is None and hasattr(analyzer, "last_X_test"):
                    st.session_state.drift_detector = DataDriftDetector(
                        reference_data=analyzer.last_X_test,
                        model_name=analyzer.last_model_name or "Best Model",
                        threshold=0.05,
                    )
                    st.session_state.reference_data = analyzer.last_X_test

                st.success(f"✅ Анализ успешно завершён! Используется {len(all_features)} признаков.")
                st.rerun()

            except Exception as e:
                logger.error(f"Критическая ошибка в блоке анализа: {str(e)}", exc_info=True)
                st.error(f"Произошла ошибка при анализе: {str(e)}")
                st.stop()

# ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ (всегда, если есть данные) ====================

if st.session_state.analysis_completed:
    # ==================== ПРЕДПРОСМОТР ЗАГРУЖЕННЫХ ДАННЫХ ====================
    df = st.session_state.df
    all_features = st.session_state.all_features
    target_col = st.session_state.get("target_column", "risk_flag")

    st.subheader("🔍 Предпросмотр загруженных сырых данных")

    df_raw = st.session_state.raw_df

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Размер таблицы:** {df_raw.shape[0]:,} строк × {df_raw.shape[1]} колонок")

    with col2:
        if st.button("Обновить предпросмотр"):
            st.rerun()

    # Информация о пропусках
    missing = df_raw.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        st.write("**Колонки с пропусками:**")
        st.dataframe(
            missing.to_frame(name="Количество пропусков").sort_values(by="Количество пропусков", ascending=False)
        )
    else:
        st.success("✅ В данных нет пропусков")

    # Показываем типы данных
    st.write("**Типы данных по колонкам:**")
    dtype_df = df_raw.dtypes.to_frame(name="Тип данных")
    dtype_df["Уникальных значений"] = df_raw.nunique()
    st.dataframe(dtype_df)

    # Показываем первые строки
    st.write("**Первые 10 строк:**")
    st.dataframe(df_raw.head(10))

    st.markdown("---")
    # Информация о данных
    st.subheader("📋 Информация о данных")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего студентов", len(df))
    with col2:
        if target_col in df.columns:
            risk_count = df[target_col].sum()
            st.metric("Целевая переменная", f"{risk_count} / {len(df)}")
    with col3:
        if target_col in df.columns:
            risk_pct = (df[target_col].sum() / len(df)) * 100
            st.metric("Процент целевого класса", f"{risk_pct:.1f}%")
    with col4:
        st.metric("Признаков", len(all_features))

    st.success(f"✅ Используется {len(all_features)} признаков")

    # Корреляционный анализ
    if st.session_state.fig_corr is not None:
        st.subheader("🔗 Корреляционный анализ")
        st.plotly_chart(st.session_state.fig_corr)
        # ==================== КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ====================
        st.subheader("🔗 Корреляционный анализ")

        # Добавь в сайдбар (в разделе "Дополнительные настройки") слайдер:
        # corr_threshold = st.slider("Порог корреляции для сильных связей", 0.0, 1.0, 0.3, 0.05)

        corr_threshold = 0.3  # можно вынести в sidebar позже

        corr_results = safe_execute(
            correlation_analysis,
            df,
            all_features,
            target_col,
            corr_threshold=corr_threshold,
            error_msg="Ошибка при расчёте корреляций",
        )

        if corr_results:
            fig_corr = px.imshow(
                corr_results["full_matrix"],
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                title=f"Корреляционная матрица (порог сильных связей: {corr_threshold})",
            )
            st.plotly_chart(fig_corr)

            # Показываем только сильные корреляции с target
            st.write("**Топ корреляций с целевой переменной:**")
            st.dataframe(corr_results["target_correlations"].head(10))

            # Кнопка сохранения графика
            if st.button("Сохранить корреляционную матрицу (PNG)"):
                save_plotly_fig(fig_corr, "correlation_matrix")
                st.success("Матрица сохранена как correlation_matrix.png")

    # Кластеризация
    st.subheader("🎯 Кластеризация студентов")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Профили кластеров:**")
        st.dataframe(st.session_state.cluster_profiles)
    with col2:
        st.plotly_chart(st.session_state.fig_clusters)

    # Выбранные признаки
    st.subheader("📊 Выбранные признаки")
    if st.session_state.get("all_features"):
        st.write(", ".join(st.session_state.all_features))
    else:
        st.write("—")
    # Кросс-валидация
    st.subheader("📊 Кросс-валидация")

    cv_results = st.session_state.get("cv_results", {})

    if cv_results and isinstance(cv_results, dict) and len(cv_results) > 0:  # Преобразуем в удобную таблицу
        cv_df = pd.DataFrame(
            {
                "Модель": list(cv_results.keys()),
                "F1 (среднее)": [cv_results[m]["mean"] for m in cv_results],
                "F1 (std)": [cv_results[m]["std"] for m in cv_results],
            }
        )
        st.dataframe(cv_df)
        st.session_state.cv_df = cv_df  # сохраняем для повторного рендера
    else:
        st.info("Результаты кросс-валидации не были получены")

    if "cv_results" not in st.session_state or not st.session_state.cv_results:
        st.session_state.cv_results = cv_results

    # Метрики на тесте
    st.subheader("📈 Метрики на тестовой выборке")
    test_metrics = st.session_state.get("test_metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F1-score", f"{test_metrics.get('f1', 0):.4f}")
    with col2:
        st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.4f}")
    with col3:
        st.metric("Precision", f"{test_metrics.get('precision', 0):.4f}")
    with col4:
        st.metric("Recall", f"{test_metrics.get('recall', 0):.4f}")

    # ROC-кривая
    if st.session_state.fig_roc:
        st.plotly_chart(st.session_state.fig_roc)

    # Confusion Matrix
    st.plotly_chart(st.session_state.fig_cm)

    # Важность признаков
    if st.session_state.fig_fi:
        st.plotly_chart(st.session_state.fig_fi)

    st.subheader("📊 Продвинутый анализ")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📈 Кросс-таблицы",
            "📁 История экспериментов",
            "📉 Временные ряды",
            "🔧 Обработка пропусков",
            "Конструктор композитных оценок",
            "Объединение признаков",
            "🔍 Выделение подмножества респондентов",
        ]
    )

    with tab1:
        st.write("### Кросс-таблицы")

        all_cols = df.columns.tolist()

        st.subheader("🔍 Настройка категориальных переменных")

        # Список паттернов для автоматического определения служебных колонок
        service_patterns = [
            r"user",
            r"id",
            r"vk",
            r"фио",
            r"фамилия",
            r"имя",
            r"отчество",
            r"пол",
            r"возраст",
            r"курс",
            r"вуз",
            r"направление",
            r"группа",
            r"date",
            r"дата",
            r"sheet",
            r"source",
            r"cluster",
            r"unnamed",
            r"index",
            r"номер",
            r"№",
        ]

        def is_likely_service(col: str) -> bool:
            col_lower = str(col).lower()
            return any(re.search(pattern, col_lower) for pattern in service_patterns)

        # Формируем рекомендуемый список на исключение
        recommended_exclude = [col for col in all_cols if is_likely_service(col)]

        # Пользовательский выбор
        exclude_cols = st.multiselect(
            "Колонки, которые **исключить** из кросс-таблиц (служебные)",
            options=all_cols,
            default=recommended_exclude,
            help="Снимите галочку с тех колонок, которые хотите оставить как категориальные",
        )

        # Выбор переменных для кросс-таблицы
        cat_threshold = st.slider(
            "Максимальное количество уникальных значений для категориальной колонки",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="cat_threshold",
        )
        # Определяем категориальные колонки
        categorical_cols = []
        for col in all_cols:
            if col in exclude_cols:
                continue
            n_unique = df[col].nunique()
            dtype = str(df[col].dtype)

            if dtype in ["object", "string"] or n_unique <= cat_threshold:
                categorical_cols.append(col)
        categorical_cols = sorted(list(set(categorical_cols)))

        st.info(f"Найдено **{len(categorical_cols)}** категориальных переменных для анализа")
        use_binning = st.checkbox(
            "Включить автоматический биннинг для числовых колонок (квартили)",
            value=True,
            help="Если включено — числовые колонки будут автоматически разбиваться на 4 группы",
        )

        # Если нет категориальных, предлагаем сгенерировать бины
        if len(categorical_cols) < 2:
            st.warning("⚠️ Недостаточно категориальных колонок для кросс-таблицы")
            st.info("Вы можете выбрать числовые колонки, они будут автоматически разбиты на группы")

            # Разрешаем любые колонки, но с предупреждением
            row_var = st.selectbox("Переменная для строк", all_cols)
            col_var = st.selectbox("Переменная для столбцов", all_cols)
            value_var = st.selectbox("Переменная для агрегации (опционально)", ["None"] + all_cols)

            if st.button("Построить кросс-таблицу"):
                with st.spinner("Построение кросс-таблицы..."):
                    from ml_core.crosstab import create_crosstab

                    df_work = df.copy()

                    # Бинаризуем числовые колонки, если нужно
                    if use_binning:
                        for var in [row_var, col_var]:
                            if (
                                var in df_work.columns
                                and df_work[var].dtype in [np.number]
                                and df_work[var].nunique() > 10
                            ):
                                # Разбиваем на квартили или децили
                                df_work[f"{var}_group"] = pd.qcut(
                                    df_work[var],
                                    q=4,
                                    labels=["Низкий", "Ниже среднего", "Выше среднего", "Высокий"],
                                    duplicates="drop",
                                )
                                if var == row_var:
                                    row_var = f"{var}_group"
                                else:
                                    col_var = f"{var}_group"

                    if value_var != "None":
                        result = create_crosstab(df_work, row_var, col_var, values=value_var, aggfunc="mean")
                    else:
                        result = create_crosstab(df_work, row_var, col_var)

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        if st.button("Экспорт таблицы в CSV"):
                            result["table"].to_csv("crosstab.csv", encoding="utf-8")
                            st.success("Сохранено как crosstab.csv")
                    with col_exp2:
                        if st.button("Экспорт таблицы в Excel"):
                            export_crosstab(result, filename="crosstab", format="excel")
                            st.success("Сохранено как crosstab.xlsx")

                    st.dataframe(result["table"])

                    if result["chi2_test"]:
                        st.write(f"**Хи-квадрат тест:** p-value = {result['chi2_test']['p_value']:.4f}")
                        st.write(
                            f"**Статистически значимо:** {'✅ Да' if result['chi2_test']['significant'] else '❌ Нет'}"
                        )

                    if result["table"].shape[0] * result["table"].shape[1] < 1000:
                        st.plotly_chart(result["heatmap"], use_container_width=True)
                        st.plotly_chart(result["stacked_bar"], use_container_width=True)
                    else:
                        st.info("Графики скрыты для производительности (слишком большая таблица)")
        else:
            # Есть категориальные колонки — нормальная работа
            row_var = st.selectbox("Переменная для строк", categorical_cols)
            col_var = st.selectbox("Переменная для столбцов", categorical_cols)
            value_var = st.selectbox("Переменная для агрегации (опционально)", ["None"] + all_cols)

            if st.button("Построить кросс-таблицу"):
                with st.spinner("Построение кросс-таблицы..."):
                    from ml_core.crosstab import create_crosstab

                    if value_var != "None":
                        result = create_crosstab(df, row_var, col_var, values=value_var, aggfunc="mean")
                    else:
                        result = create_crosstab(df, row_var, col_var)

                    st.dataframe(result["table"])

                    if result["chi2_test"]:
                        st.write(f"**Хи-квадрат тест:** p-value = {result['chi2_test']['p_value']:.4f}")
                        st.write(
                            f"**Статистически значимо:** {'✅ Да' if result['chi2_test']['significant'] else '❌ Нет'}"
                        )

                    if result["table"].shape[0] * result["table"].shape[1] < 1000:
                        st.plotly_chart(result["heatmap"], use_container_width=True)
                        st.plotly_chart(result["stacked_bar"], use_container_width=True)
                    else:
                        st.info("Графики скрыты для производительности (слишком большая таблица)")
    with tab2:
        st.write("### История экспериментов")

        from ml_core.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()
        experiments = tracker.list_experiments()

        if not experiments.empty:
            st.dataframe(experiments)

            selected_exp = st.selectbox("Выберите эксперимент для загрузки", experiments["id"].tolist())
            if st.button("Загрузить эксперимент"):
                exp_data = tracker.load_experiment(selected_exp)
                st.json(exp_data["metrics"])
                if "model" in exp_data:
                    st.success(f"Модель загружена: {exp_data['model_name']}")
        else:
            st.info("Нет сохраненных экспериментов")

        # Кнопка сохранения текущего эксперимента
        exp_name = st.text_input("Название эксперимента")
        if st.button("Сохранить текущий анализ") and exp_name:
            exp_data = {
                "metrics": test_metrics,
                "features": st.session_state.selected_cols,
                "model_name": st.session_state.best_name if st.session_state.best_name else "unknown",
                "n_samples": len(st.session_state.df) if st.session_state.df is not None else 0,
                "description": exp_name,
            }
            exp_id = tracker.save_experiment(exp_name, exp_data)
            st.success(f"Эксперимент сохранен: {exp_id}")
    with tab3:
        st.write("### Анализ временных рядов")

        from ml_core.timeseries import detect_negative_dynamics, analyze_student_trajectory

        # Проверяем наличие необходимых колонок
        if "semester" not in df.columns:
            st.warning("⚠️ Для анализа временных рядов нужна колонка 'semester'")
            st.info("Вы можете добавить семестр в данные или использовать синтетическую генерацию с semester")

            # Предлагаем сгенерировать семестры
            if st.button("Сгенерировать тестовые семестры"):
                # Добавляем искусственные семестры
                df["semester"] = np.random.choice([1, 2, 3], size=len(df), p=[0.4, 0.3, 0.3])
                st.success("Добавлена колонка 'semester' с тестовыми значениями")
                st.rerun()
        else:
            # Проверяем наличие колонки для анализа
            available_metrics = [c for c in df.columns if c not in ["student_id", "semester", "cluster", target_col]]
            numeric_metrics = [c for c in available_metrics if df[c].dtype in [np.number]]

            if not numeric_metrics:
                st.warning("Нет числовых колонок для анализа динамики")
            else:
                value_col = st.selectbox("Показатель для анализа", numeric_metrics)

                if st.button("Выявить студентов с отрицательной динамикой"):
                    with st.spinner("Анализ..."):
                        dynamics = detect_negative_dynamics(
                            df, student_id_col="student_id", time_col="semester", value_col=value_col, threshold=-0.05
                        )

                        if "error" in dynamics:
                            st.error(dynamics["error"])
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Проанализировано студентов", dynamics["n_students_analyzed"])
                            with col2:
                                st.metric("Студентов с отрицательной динамикой", len(dynamics["at_risk_students"]))
                                st.metric("Процент риска по динамике", f"{dynamics['risk_percentage']:.1f}%")

                            if not dynamics["at_risk_students"].empty:
                                st.write("**Студенты с отрицательной динамикой:**")
                                st.dataframe(
                                    dynamics["at_risk_students"][["student_id", "trend", "first_value", "last_value"]]
                                )

                                # Выбор студента для детального анализа
                                student_id = st.selectbox(
                                    "Выберите студента для детального анализа",
                                    dynamics["at_risk_students"]["student_id"].tolist(),
                                )
                                if st.button("Показать траекторию"):
                                    trajectory = analyze_student_trajectory(
                                        df, student_id, time_col="semester", value_col=value_col
                                    )
                                    if trajectory["figure"]:
                                        st.plotly_chart(trajectory["figure"])
                                    st.write(f"**Тренд:** {trajectory['trend']:.3f}")
                                    st.write(f"**Статус:** {trajectory['status']}")
                            else:
                                st.success("✅ Студентов с отрицательной динамикой не обнаружено")
                if st.button("Показать траекторию"):
                    trajectory = safe_execute(analyze_student_trajectory, df, student_id, value_col=value_col)
                    if trajectory and "figure" in trajectory:
                        st.plotly_chart(trajectory["figure"])

                # Новый блок прогноза
                if st.button("Прогнозировать на следующие семестры"):
                    forecast = safe_execute(forecast_grades, df, student_id, value_col=value_col, future_semesters=2)
                    if forecast:
                        st.write("**Прогноз на будущие семестры:**")
                        for sem, pred in zip(forecast["future_semesters"], forecast["predictions"]):
                            st.write(f"Семестр {sem}: {pred:.2f}")
    with tab4:
        st.write("### Обработка пропусков")

        from ml_core.imputation import handle_missing_values, detect_outliers

        # Показываем текущие пропуски
        missing_before = df.isna().sum().sum()
        st.write(f"**Пропусков до обработки:** {missing_before}")
        st.dataframe(df.isna().sum().to_frame("Пропуски"))
        if missing_before > 0:
            st.dataframe(df.isna().sum().to_frame("Пропуски").sort_values(by="Пропуски", ascending=False))
        else:
            st.success("✅ В данных нет пропусков")

        # Выбросы
        outliers = detect_outliers(df)
        if any(report["n_outliers"] > 0 for report in outliers.values()):
            st.write("### Выбросы в данных")
            for col, report in outliers.items():
                if report["n_outliers"] > 0:
                    st.write(f"**{col}:** {report['n_outliers']} выбросов ({report['percentage']:.1f}%)")

        strategy = st.selectbox("Стратегия обработки", ["auto", "fill_median", "fill_mean", "interpolate", "drop_rows"])

        if st.button("Применить обработку"):
            df_clean, report = handle_missing_values(df, strategy=strategy)
            st.write("### Отчет об обработке")
            st.write(f"**Было:** {report['original_shape']}")
            st.write(f"**Стало:** {report['final_shape']}")
            st.write(f"**Удалено пропусков:** {report['missing_before'] - report['missing_after']}")

            for action in report["actions"]:
                st.write(f"- {action['column']}: {action['message']}")

            # Обновляем df в session_state
            st.session_state.df = df_clean
            st.success("Данные обновлены! Перезапустите анализ для применения изменений.")
    with tab5:
        st.subheader("### Конструктор композитных оценок")

        if "df" not in st.session_state or st.session_state.df is None:
            st.warning("Сначала загрузите данные и выполните анализ")
            st.stop()

        current_df = st.session_state.df

        # === Улучшенная фильтрация: исключаем служебные колонки ===
        service_patterns = [
            r"^user",
            r"^user_id",
            r"^vk",
            r"^VK",
            r"^id",
            r"^index",
            r"фамилия",
            r"имя",
            r"отчество",
            r"фио",
            r"вуз",
            r"направление",
            r"пол",
            r"возраст",
            r"курс",
            r"группа",
            r"дата",
            r"date",
            r"sheet",
            r"source",
            r"unnamed",
            r"cluster",
            r"risk_flag",
            r"risk_pred",
            r"prediction",
        ]

        def is_service_column(col: str) -> bool:
            col_lower = str(col).lower()
            return any(re.search(pattern, col_lower) for pattern in service_patterns)

        # Берем только числовые колонки, которые НЕ являются служебными
        numeric_features = [
            col
            for col in current_df.columns
            if current_df[col].dtype.kind in "biufc"  # числовые типы
            and not is_service_column(col)
            and current_df[col].nunique() > 1  # исключаем константы
        ]

        if not numeric_features:
            st.error("В данных не найдено подходящих числовых признаков для создания композитной оценки")
            st.stop()

        st.write(f"Доступно **{len(numeric_features)}** числовых признаков для построения композитной оценки")

        # === Интерфейс задания весов ===
        weights = {}
        cols_layout = st.columns(3)

        for i, feat in enumerate(numeric_features):
            with cols_layout[i % 3]:
                weight = st.slider(
                    label=f"{feat}", min_value=-3.0, max_value=3.0, value=0.0, step=0.05, key=f"weight_{feat}"
                )
                weights[feat] = weight

        score_name = st.text_input(
            "Название новой композитной оценки", value="custom_risk_index", key="score_name_input"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Создать композитную оценку", type="primary", use_container_width=True):
                active_weights = {k: v for k, v in weights.items() if abs(v) > 1e-6}

                if not active_weights:
                    st.error("❌ Укажите хотя бы один ненулевой вес")
                else:
                    with st.spinner("Создаётся композитная оценка..."):
                        try:
                            df_new, new_col = build_composite_score(
                                current_df, active_weights, score_name=score_name, assign_id=True
                            )

                            st.success(f"✅ Создана оценка **{new_col}**")
                            st.dataframe(df_new[[new_col]].describe().round(4))

                            st.write("**Использованные веса:**")
                            st.json(active_weights)

                            # Обновляем данные
                            st.session_state.df = df_new
                            if new_col not in st.session_state.get("all_features", []):
                                if "all_features" not in st.session_state:
                                    st.session_state.all_features = []
                                st.session_state.all_features.append(new_col)

                            st.rerun()

                        except Exception as e:
                            st.error(f"Ошибка при создании оценки: {str(e)}")
                            logger.error(f"build_composite_score error: {e}", exc_info=True)

        with col2:
            if st.button("Сбросить все веса"):
                st.rerun()

    with tab6:
        st.subheader("🔗 Объединение признаков и создание связей")

        col1, col2 = st.columns(2)

        with col1:
            numerical_cols = st.multiselect(
                "Числовые признаки для комбинаций",
                options=[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                default=[],
            )

        with col2:
            text_cols = st.multiselect(
                "Текстовые признаки", options=[col for col in df.columns if df[col].dtype == "object"], default=[]
            )

        max_pairs = st.slider("Максимальное количество новых признаков", 5, 30, 15)

        if st.button("Создать комбинации признаков", type="primary"):
            with st.spinner("Генерация новых признаков..."):
                from ml_core.features import create_feature_combinations

                df_new = create_feature_combinations(
                    df, numerical_cols=numerical_cols, text_cols=text_cols, max_pairs=max_pairs
                )

                st.success(f"Создано {len(df_new.columns) - len(df.columns)} новых признаков")
                st.dataframe(df_new.head())

                # Обновляем данные в session_state
                st.session_state.df = df_new
    with tab7:
        st.subheader("🔍 Выделение подмножества респондентов")

        if "df" not in st.session_state or st.session_state.df is None:
            st.warning("Сначала загрузите данные и выполните анализ")
        else:
            df = st.session_state.df
            method = st.radio(
                "Способ выбора подмножества", ["Произвольная выборка", "По условию (query)", "По кластеру"]
            )
            if method == "Произвольная выборка":
                n_samples = st.slider(
                    "Количество респондентов", min_value=10, max_value=len(df), value=min(100, len(df))
                )

                if st.button("Выделить случайную выборку", type="primary"):
                    with st.spinner("Выполняется выборка..."):
                        subset = analyzer.select_subset(df, n_samples=n_samples)
                        st.success(f"Выбрано {len(subset)} респондентов")
                        st.dataframe(subset)

            elif method == "По условию (query)":
                condition = st.text_input("Условие (pandas query)", placeholder="avg_grade < 3.0 and stress_level > 7")
                if st.button("Применить фильтр", type="primary"):
                    if not condition:
                        st.error("Введите условие фильтрации")
                    else:
                        with st.spinner("Применяем фильтр..."):
                            try:
                                subset = analyzer.select_subset(df, condition=condition)
                                st.success(f"Выбрано {len(subset)} респондентов")
                                st.dataframe(subset)
                            except Exception as e:
                                st.error(f"Ошибка в условии: {e}")

            elif method == "По кластеру":
                if "cluster" not in df.columns:
                    st.warning("Сначала выполните кластеризацию студентов (в основном анализе)")
                else:
                    cluster_id = st.selectbox("Выберите номер кластера", sorted(df["cluster"].unique()))
                    if st.button("Показать кластер", type="primary"):
                        with st.spinner("Загрузка кластера..."):
                            subset = analyzer.select_subset(df, by_cluster=cluster_id)
                            st.success(f"Кластер {cluster_id} — {len(subset)} студентов")
                            st.dataframe(subset)
    # SHAP объяснения
    st.subheader("💡 Объяснения предсказаний")
    if st.session_state.explanations:
        for exp in st.session_state.explanations[:3]:
            with st.expander(f"Студент #{exp['student_index']} - Риск: {exp['risk_probability']:.1%}"):
                st.markdown(exp["explanation"])
    else:
        st.info("SHAP объяснения не сгенерированы")

# ==================== ЭКСПОРТ (всегда внизу) ====================
st.subheader("💾 Экспорт результатов")

if st.session_state.analysis_completed:
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.X_test_sel is not None:
            predictions_df = pd.DataFrame(st.session_state.X_test_sel, columns=st.session_state.selected_cols)
            predictions_df["risk_pred"] = st.session_state.y_pred
            predictions_df["risk_probability"] = st.session_state.best_model.predict_proba(st.session_state.X_test_sel)[
                :, 1
            ]
            csv = predictions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Скачать предсказания (CSV)",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True,
                key="download_pred",
            )

    with col2:
        if st.session_state.explanations is not None:
            explanations_df = pd.DataFrame(
                [
                    {
                        "student_index": e["student_index"],
                        "risk_probability": e["risk_probability"],
                        "risk_level": e["risk_level"],
                        "explanation": e["explanation"],
                    }
                    for e in st.session_state.explanations
                ]
            )
            csv_exp = explanations_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Скачать объяснения (CSV)",
                csv_exp,
                "explanations.csv",
                "text/csv",
                use_container_width=True,
                key="download_exp",
            )

    st.caption("✅ Результаты последнего анализа сохранены. Можете скачивать файлы.")
else:
    if not st.session_state.data_loaded:
        st.info("👈 Загрузите данные в левой панели, чтобы начать анализ.")

# После экспорта, добавьте:
st.markdown("---")
st.subheader("📊 Мониторинг модели")

if st.session_state.analysis_completed and st.session_state.drift_detector:
    tab1, tab2, tab3 = st.tabs(["📈 Дрейф данных", "📉 История метрик", "⚙️ Настройки мониторинга"])

    with tab1:
        st.write("### Проверка дрейфа на новых данных")

        # Загрузка новых данных для проверки дрейфа
        new_data_file = st.file_uploader(
            "Загрузите новые данные для проверки дрейфа",
            type=["csv", "xlsx", "xls"],
            help="Поддерживаются форматы: CSV, Excel (.xlsx, .xls)",
            key="drift_upload",
        )

        use_full_reference = st.checkbox("Использовать полный датасет как эталон", value=True)

        if new_data_file is not None:
            file_extension = new_data_file.name.split(".")[-1].lower()
            new_df = None
            sheet_name = None

            try:
                if file_extension == "csv":
                    new_df = pd.read_csv(new_data_file)
                    st.info(f"Загружен CSV файл: {new_data_file.name}")

                else:  # Excel файл
                    # Получаем список листов
                    xl = pd.ExcelFile(new_data_file)
                    sheet_names = xl.sheet_names

                    if len(sheet_names) > 1:
                        sheet_name = st.selectbox(
                            "Выберите лист для проверки дрейфа", options=sheet_names, key="drift_sheet_selector"
                        )
                    else:
                        sheet_name = sheet_names[0]

                    # Загружаем выбранный лист
                    new_df = pd.read_excel(new_data_file, sheet_name=sheet_name)
                    st.info(f"Загружен Excel файл: {new_data_file.name} (лист: {sheet_name})")

                # Проверяем наличие нужных признаков
                current_cols = st.session_state.get("selected_cols", [])

                if not current_cols:
                    st.warning("Сначала выполните полный анализ, чтобы определить нужные признаки")
                elif all(col in new_df.columns for col in current_cols):
                    # Заполняем пропуски только по числовым колонкам
                    new_data = new_df[current_cols].copy()
                    numeric_cols = new_data.select_dtypes(include=[np.number]).columns
                    new_data[numeric_cols] = new_data[numeric_cols].fillna(new_data[numeric_cols].median())

                    if st.button("🔍 Проверить дрейф", type="primary", use_container_width=True):
                        with st.spinner("Анализ дрейфа данных..."):

                            if use_full_reference and st.session_state.get("X_train_sel") is not None:
                                full_data = pd.DataFrame(
                                    np.vstack([st.session_state.X_train_sel, st.session_state.X_test_sel]),
                                    columns=st.session_state.selected_cols,
                                )
                                st.session_state.drift_detector.reference_data = full_data
                            else:
                                st.session_state.drift_detector.reference_data = st.session_state.reference_data

                            drift_report = st.session_state.drift_detector.detect_drift(new_data)
                            st.session_state.drift_report = drift_report

                            # Отображение результатов
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Дрейфующих признаков", len(drift_report.get("drifted_features", [])))
                            with col2:
                                st.metric("Процент дрейфа", f"{drift_report.get('drift_percentage', 0):.1f}%")
                            with col3:
                                status = "🔴 Есть дрейф" if drift_report.get("overall_drift") else "🟢 Нет дрейфа"
                                st.metric("Статус", status)

                            if drift_report.get("overall_drift"):
                                st.warning(st.session_state.drift_detector.generate_alert_message(drift_report))
                            else:
                                st.success("✅ Данные стабильны, модель можно использовать")

                            with st.expander("Детальный отчет о дрейфе"):
                                st.json(drift_report)

                            report_path = st.session_state.drift_detector.save_report(drift_report)
                            st.info(f"Отчет сохранён: {report_path}")

                else:
                    missing = set(current_cols) - set(new_df.columns)
                    st.error(f"В загруженных данных отсутствуют необходимые признаки: {missing}")

            except Exception as e:
                st.error(f"Ошибка при чтении файла: {str(e)}")
    with tab2:
        st.write("### История метрик модели")

        # Показываем историю метрик из логов
        if os.path.exists("logs/model_metrics.csv"):
            history_df = pd.read_csv("logs/model_metrics.csv")
            st.dataframe(history_df)

            # График изменения F1-score
            if len(history_df) > 1:
                fig_history = px.line(history_df, x="timestamp", y="f1_score", title="Динамика качества модели")
                st.plotly_chart(fig_history)
        else:
            st.info("История метрик пока пуста")
    with tab3:
        st.write("### Настройки мониторинга")

        new_threshold = st.slider(
            "Порог чувствительности дрейфа (p-value)",
            min_value=0.01,
            max_value=0.1,
            value=0.05,
            step=0.01,
            help="Чем меньше значение, тем чувствительнее детектор",
        )

        if st.button("Обновить порог"):
            st.session_state.drift_detector.threshold = new_threshold
            st.success(f"Порог обновлен на {new_threshold}")

        # Кнопка для переобучения
        if st.button("🔄 Запланировать переобучение"):
            st.info("Функция переобучения будет доступна в следующей версии")
