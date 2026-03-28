import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
# Импортируем ВАШИ модули
from ml_core.data import load_from_csv
from ml_core.features import add_composite_features, get_base_features, select_features_for_model, preprocess_data
from ml_core.analysis import (
    correlation_analysis, cluster_students,
    analyze_cluster_profiles, plot_clusters_pca
)
from ml_core.llm_interface import LLMInterface
from ml_core.models import ModelTrainer
from ml_core.evaluation import (
    plot_roc_curves, plot_confusion_matrix,
    plot_feature_importance, generate_shap_explanations
)
import plotly.graph_objects as go
from monitoring.logger import MLLogger
from monitoring.drift_detector import DataDriftDetector, DriftMonitorThread, DriftMonitorScheduler
from ml_core.analysis import correlation_analysis_enhanced
from ml_core.features import build_composite_score
from ml_core.timeseries import forecast_grades
from ml_core.crosstab import export_crosstab
from ml_core.error_handler import safe_execute
from ml_core.analysis import save_plotly_fig


def plot_drift_visualization(drift_report, reference_data, current_data):
    """
    Визуализация дрейфа для ключевых признаков
    """
    import plotly.subplots as sp

    # Берем топ-4 признака с наибольшим дрейфом
    drifted_features = drift_report['drifted_features'][:4]

    if not drifted_features:
        return None

    # Создаем подграфики
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=drifted_features,
        vertical_spacing=0.15
    )

    for i, feature in enumerate(drifted_features):
        row = i // 2 + 1
        col = i % 2 + 1

        # Добавляем распределение эталонных данных
        fig.add_trace(
            go.Histogram(
                x=reference_data[feature],
                name='Эталон',
                opacity=0.7,
                marker_color='blue',
                nbinsx=30
            ),
            row=row, col=col
        )

        # Добавляем распределение новых данных
        fig.add_trace(
            go.Histogram(
                x=current_data[feature],
                name='Новые данные',
                opacity=0.7,
                marker_color='red',
                nbinsx=30
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=600,
        title_text="Сравнение распределений признаков",
        showlegend=True
    )

    return fig


# Инициализация session_state
if 'results_saved' not in st.session_state:
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
    st.session_state.target_column = 'risk_flag'
    st.session_state.data_category = 'grades'
    st.session_state.target_selected = False  # Флаг выбора целевой переменной
    st.session_state.data_loaded = False  # Флаг загрузки данных
    st.session_state.raw_df = None  # Исходные данные без обработки

    if 'drift_detector' not in st.session_state:
        st.session_state.drift_detector = None
        st.session_state.drift_report = None
        st.session_state.reference_data = None
        if 'monitor_thread' not in st.session_state:
            st.session_state.monitor_thread = None

# Инициализация логгера
ml_logger = MLLogger()

# Настройка страницы
st.set_page_config(
    page_title="Система мониторинга академических рисков",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Система мониторинга академических рисков студентов")
st.markdown("---")

# Инициализация трейнера моделей
trainer = ModelTrainer(models_dir='models')

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")

    # Загрузка данных
    st.subheader("📁 Загрузка данных")

    data_type = st.radio(
        "Тип данных",
        ["Синтетические данные", "CSV файл", "Excel файл"],
        key="data_type_radio"
    )

    df = None
    uploaded_file = None
    excel_file = None

    if data_type == "Синтетические данные":
        st.session_state.synthetic_category = st.selectbox(
            "Категория данных",
            ['grades', 'psychology', 'creativity', 'values', 'personality', 'activities', 'career'],
            format_func=lambda x: {
                'grades': '📚 Успеваемость (риск отчисления)',
                'psychology': '🧠 Психологическое состояние (риск выгорания)',
                'creativity': '🎨 Креативность (высокая креативность)',
                'values': '💎 Ценности Шварца (тип профиля)',
                'personality': '👤 Личностные опросы (потенциал лидерства)',
                'activities': '⚡ Активность (активный участник)',
                'career': '💼 Карьерные намерения (определённость)'
            }.get(x, x)
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

                result = load_data(
                    category=category,
                    n_students=n_students,
                    generate_two_sets=generate_drift_data
                )

                st.session_state.raw_df = result['data']
                st.session_state.data_category = category
                st.session_state.target_column = result['target']
                st.session_state.data_loaded = True
                st.session_state.target_selected = True  # Для синтетики цель уже выбрана
                st.session_state.analysis_completed = False  # Сбрасываем анализ

                if generate_drift_data and 'reference' in result and 'new' in result:
                    st.session_state.drift_reference = result['reference']
                    st.session_state.drift_new_data = result['new']

                st.success(f"✅ Синтетические данные загружены ({len(result['data'])} записей)")
                st.rerun()

    elif data_type == "CSV файл":
        uploaded_file = st.file_uploader(
            "Выберите CSV файл",
            type=['csv'],
            help="Файл должен содержать колонки с признаками и целевую переменную 'risk_flag'",
            key="csv_uploader"
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
            "Выберите Excel файл",
            type=['xlsx', 'xls'],
            help="Файл с опросом/опросами",
            key="excel_uploader"
        )
        if excel_file is not None:
            from ml_core.loader import get_sheet_names, load_excel_sheet

            # Сохраняем временно
            with open('temp_excel.xlsx', 'wb') as f:
                f.write(excel_file.getbuffer())

            sheet_names = get_sheet_names('temp_excel.xlsx')

            if 'selected_sheet' not in st.session_state:
                st.session_state.selected_sheet = sheet_names[0] if sheet_names else None

            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Выберите лист для анализа",
                    sheet_names,
                    index=sheet_names.index(
                        st.session_state.selected_sheet) if st.session_state.selected_sheet in sheet_names else 0,
                    key="sheet_selector"
                )
                st.session_state.selected_sheet = selected_sheet
            else:
                selected_sheet = sheet_names[0]

            if st.button("📂 Загрузить выбранный лист", key="load_sheet"):
                df, msg = load_excel_sheet('temp_excel.xlsx', selected_sheet)
                if df is not None:
                    st.session_state.raw_df = df
                    st.session_state.data_loaded = True
                    st.session_state.target_selected = False
                    st.session_state.analysis_completed = False
                    st.session_state.data_category = 'excel'
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
        optimization_metric = st.selectbox(
            "Метрика для оптимизации",
            ["f1", "roc_auc", "precision", "recall"],
            index=0
        )

        corr_threshold = st.slider("Порог корреляции для сильных связей", 0.0, 1.0, 0.3, 0.05)
        # Количество признаков для отбора
        n_features_to_select = st.slider(
            "Количество признаков в финальной модели",
            min_value=3,
            max_value=15,
            value=7
        )

        # Выбор моделей для сравнения
        st.write("**Модели для обучения:**")
        use_lr = st.checkbox("Logistic Regression", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_xgb = st.checkbox("XGBoost", value=True)

        # Порог для SHAP-объяснений
        shap_top_n = st.slider(
            "Количество факторов в объяснениях",
            min_value=3,
            max_value=10,
            value=5
        )

        # Кнопка запуска
        run_analysis = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

# ==================== ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ====================
# Отображаем выбор целевой переменной, если данные загружены и цель не выбрана
if st.session_state.data_loaded and not st.session_state.target_selected:
    st.subheader("🎯 Выбор целевой переменной")

    df_raw = st.session_state.raw_df

    # Исключаем служебные колонки
    exclude = ['user', 'user_id', 'VK_id', 'дата', 'date', '_source_sheet', '_sheet_type']
    candidate_cols = [col for col in df_raw.columns if col not in exclude]

    if not candidate_cols:
        st.error("Нет подходящих колонок для целевой переменной")
        st.stop()

    # Показываем информацию о данных
    st.write(f"**Всего записей:** {len(df_raw)}")
    st.write(
        f"**Колонки в данных:** {', '.join(df_raw.columns.tolist()[:10])}{'...' if len(df_raw.columns) > 10 else ''}")

    # Выбор целевой переменной
    target_choice = st.selectbox(
        "Колонка для предсказания (целевая переменная)",
        candidate_cols,
        help="Можно выбрать любую колонку – система построит модель предсказания для неё."
    )

    # Показываем статистику по выбранной колонке
    if target_choice in df_raw.columns:
        col_data = df_raw[target_choice]
        st.write(f"**Статистика выбранной колонки:**")
        st.write(f"- Тип данных: {col_data.dtype}")
        st.write(f"- Уникальных значений: {col_data.nunique()}")

        if col_data.dtype in ['int64', 'float64']:
            st.write(f"- Диапазон: {col_data.min():.2f} - {col_data.max():.2f}")
            # Предлагаем преобразовать в бинарную переменную
            if col_data.nunique() > 2:
                st.info(
                    f"Колонка содержит {col_data.nunique()} уникальных значений. Будет преобразована в бинарную (порог = медиана)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Подтвердить выбор цели", type="primary", use_container_width=True):
            st.session_state.target_column = target_choice
            st.session_state.target_selected = True
            st.session_state.data_category = 'custom'
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
    run_analysis_flag = st.sidebar.button("🚀 Запустить анализ", type="primary",
                                          use_container_width=True) if 'run_analysis' not in locals() else run_analysis

    if run_analysis_flag or not st.session_state.analysis_completed:
        # Логируем запуск
        ml_logger.log_event("analysis_started", {
            "data_type": data_type,
            "n_clusters": n_clusters,
            "use_hp_tuning": use_hp_tuning
        })

        with st.spinner("Загрузка и обработка данных..."):
            df = st.session_state.raw_df.copy()
            target_col = st.session_state.target_column
            category = st.session_state.data_category

            # Определяем базовые признаки
            base_features = get_base_features(df)
            df = add_composite_features(df)

            # Добавляем композитные признаки
            composite_features = ['trend_grades', 'grade_stability', 'cognitive_load',
                                  'overall_satisfaction', 'psychological_wellbeing', 'academic_activity']
            all_features = base_features + [f for f in composite_features if f in df.columns]

            # Логируем признаки
            ml_logger.log_features(all_features)

            # Создаём целевую переменную, если её нет
            if target_col not in df.columns:
                st.warning(f"Целевая переменная '{target_col}' не найдена, создаем на основе данных")

                if category == 'grades':
                    if 'avg_grade' in df.columns:
                        df[target_col] = (df['avg_grade'] < 3.0).astype(int)
                    elif 'Сумма' in df.columns:
                        df[target_col] = (df['Сумма'] < 80).astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.3, size=len(df))

                elif category == 'psychology':
                    if 'stress_level' in df.columns and 'motivation_score' in df.columns:
                        burnout = (df['stress_level'] > 7).astype(int) * 0.5 + (df['motivation_score'] < 4).astype(
                            int) * 0.5
                        df[target_col] = (burnout > 0.5).astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.3, size=len(df))

                elif category == 'creativity':
                    if 'creativity_total' in df.columns:
                        df[target_col] = (df['creativity_total'] > 100).astype(int)
                    elif 'Сумма' in df.columns:
                        df[target_col] = (df['Сумма'] > 100).astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.3, size=len(df))

                elif category == 'values':
                    # Для ценностей создаём профиль (открытый vs консервативный)
                    if all(c in df.columns for c in ['self_direction', 'stimulation', 'security', 'conformity']):
                        open_score = df['self_direction'] + df['stimulation']
                        conservative_score = df['security'] + df['conformity']
                        df[target_col] = (open_score > conservative_score).astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.5, size=len(df))

                elif category == 'personality':
                    if 'leadership' in df.columns:
                        df[target_col] = (df['leadership'] > 3).astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.4, size=len(df))

                elif category == 'activities':
                    if 'activity_score' in df.columns:
                        df[target_col] = (df['activity_score'] >= 3).astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.3, size=len(df))

                elif category == 'career':
                    if 'work_by_specialty' in df.columns:
                        df[target_col] = (df['work_by_specialty'] == 'Да').astype(int)
                    else:
                        df[target_col] = np.random.binomial(1, 0.5, size=len(df))

                else:
                    # Для пользовательских данных пробуем преобразовать в бинарную
                    if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 2:
                        median_val = df[target_col].median()
                        df[target_col] = (df[target_col] > median_val).astype(int)
                        st.info(f"Колонка '{target_col}' преобразована в бинарную (порог = медиана {median_val:.2f})")
                    else:
                        df[target_col] = np.random.binomial(1, 0.3, size=len(df))
            else:
                st.info(f"✅ Целевая переменная '{target_col}' уже присутствует в данных")

                # Проверяем, что колонка только одна
                if list(df.columns).count(target_col) > 1:
                    # Оставляем только первую колонку
                    cols_to_keep = []
                    first_found = False
                    for col in df.columns:
                        if col == target_col:
                            if not first_found:
                                cols_to_keep.append(col)
                                first_found = True
                        else:
                            cols_to_keep.append(col)
                    df = df[cols_to_keep]
                    st.warning(f"⚠️ Удалены дублирующиеся колонки '{target_col}'")

                # Преобразуем в бинарную если нужно
                if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 2:
                    median_val = df[target_col].median()
                    df[target_col] = (df[target_col] > median_val).astype(int)
                    st.info(f"Колонка '{target_col}' преобразована в бинарную (порог = медиана {median_val:.2f})")

            # Удаляем дубликаты колонок
            duplicate_cols = [col for col in df.columns if list(df.columns).count(col) > 1]
            if duplicate_cols:
                st.warning(f"⚠️ Найдены дублирующиеся колонки: {set(duplicate_cols)}. Удаляем...")
                df = df.loc[:, ~df.columns.duplicated()]
                st.success("✅ Дубликаты колонок удалены")

            # Корреляционный анализ
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in df.columns and len(numeric_cols) > 0:
                if target_col not in numeric_cols:
                    numeric_cols.append(target_col)

                corr = correlation_analysis(df, numeric_cols, target_col)
                if corr.columns.duplicated().any():
                    corr = corr.loc[:, ~corr.columns.duplicated()]

                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="Корреляционная матрица"
                )
                st.session_state.fig_corr = fig_corr
            else:
                st.session_state.fig_corr = None

            # Кластеризация
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Подготовка данных...")
            # Берем только числовые колонки для кластеризации
            cluster_features = [c for c in all_features if c in df.columns and df[c].dtype in [np.number]]
            if len(cluster_features) < 2:
                cluster_features = numeric_cols[:min(5, len(numeric_cols))]

            X_cluster = df[all_features].fillna(df[all_features].median())
            progress_bar.progress(25)

            status_text.text("Масштабирование признаков...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            progress_bar.progress(50)

            status_text.text("Выполнение кластеризации...")
            cluster_labels, kmeans_model, scaler_cluster = cluster_students(
                X_cluster,
                n_clusters=n_clusters,
                feature_cols=all_features
            )
            df['cluster'] = cluster_labels
            progress_bar.progress(75)

            status_text.text("Анализ профилей...")
            cluster_profiles = analyze_cluster_profiles(df, all_features)
            fig_clusters = plot_clusters_pca(
                X_cluster,
                cluster_labels,
                all_features)
            progress_bar.progress(100)

            progress_bar.empty()
            status_text.empty()

            # Подготовка к обучению
            if target_col in df.columns:
                model_features = [c for c in all_features if c in df.columns and df[c].dtype in [np.number]]
                if len(model_features) < 2:
                    model_features = numeric_cols

                X_resampled, y_resampled = preprocess_data(df, all_features, target_col)

                # Разделяем сбалансированные данные
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_resampled
                )

                # Отбор признаков
                X_train_sel, selected_cols = select_features_for_model(
                    X_train, y_train,
                    final_n=n_features_to_select
                )
                X_test_sel = X_test[selected_cols]

                # Кросс-валидация
                cv_results = trainer.cross_validate(X_train_sel, y_train)
                cv_df = pd.DataFrame({
                    'Модель': list(cv_results.keys()),
                    'F1 (среднее)': [cv_results[m]['mean'] for m in cv_results],
                    'F1 (std)': [cv_results[m]['std'] for m in cv_results]
                })

                # Обучение лучшей модели
                best_model, best_name, metrics = trainer.train_best_model(
                    X_train_sel, y_train, X_test_sel, y_test
                )
                models_to_train = {}
                if use_lr:
                    models_to_train['LR'] = LogisticRegression(max_iter=1000)
                if use_rf:
                    models_to_train['RF'] = RandomForestClassifier(random_state=42)
                if use_xgb:
                    if use_hp_tuning:
                        st.subheader("⚙️ Оптимизация гиперпараметров XGBoost")
                        with st.spinner("Оптимизация гиперпараметров..."):
                            xgb_tuned, best_params, best_cv_score = trainer.tune_xgboost(
                                X_train_sel,
                                y_train,
                                n_iter=n_iter_tuning,
                                cv_folds=3
                            )
                            st.success(f"✅ Оптимизация завершена. Лучший F1-score: {best_cv_score:.4f}")
                            st.json(best_params)
                            models_to_train['XGB'] = xgb_tuned
                    else:
                        models_to_train['XGB'] = XGBClassifier(eval_metric='logloss', random_state=42)

                trainer.models = models_to_train

                # Сохраняем модель
                model_path, metadata = trainer.save_model(
                    best_model, best_name, metrics, selected_cols
                )
                st.info(f"💾 Модель сохранена: {model_path}")

                # Логируем результаты
                ml_logger.log_model_metrics(best_name, metrics)

                # Метрики на тесте
                test_metrics = metrics['test']

                # ROC-кривая
                models_for_plot = {}
                if use_lr and 'LR' in trainer.models:
                    models_for_plot['LR'] = trainer.models['LR']
                if use_rf and 'RF' in trainer.models:
                    models_for_plot['RF'] = trainer.models['RF']
                if use_xgb and 'XGB' in trainer.models:
                    models_for_plot['XGB'] = trainer.models['XGB']

                for name, model in models_for_plot.items():
                    model.fit(X_train_sel, y_train)

                if len(models_for_plot) > 0:
                    fig_roc = plot_roc_curves(models_for_plot, X_test_sel, y_test)
                else:
                    fig_roc = None

                # Confusion Matrix
                y_pred = best_model.predict(X_test_sel)
                fig_cm = plot_confusion_matrix(y_test, y_pred, best_name)

                # Важность признаков
                fig_fi = plot_feature_importance(best_model, selected_cols)

                # После того как получили best_model и selected_cols
                if st.session_state.drift_detector is None:
                    # Отбираем нужные признаки
                    reference_real = df[selected_cols].copy()

                    st.session_state.drift_detector = DataDriftDetector(
                        reference_data=reference_real,
                        model_name=best_name,
                        threshold=0.05,
                        model_metadata={
                            'numerical_features': selected_cols,
                            'categorical_features': [],
                            'baseline_metrics': test_metrics
                        }
                    )
                    st.session_state.reference_data = reference_real

                # SHAP объяснения
                X_test_df = pd.DataFrame(X_test_sel, columns=selected_cols)
                explanations = generate_shap_explanations(
                    best_model,
                    X_test_df,
                    selected_cols,
                    threshold=risk_threshold,
                    top_n=shap_top_n
                )

                # ✅ СОХРАНЯЕМ ВСЁ В SESSION_STATE
                st.session_state.analysis_completed = True
                st.session_state.results_saved = True
                st.session_state.df = df
                st.session_state.all_features = all_features
                st.session_state.cluster_labels = cluster_labels
                st.session_state.cluster_profiles = cluster_profiles
                st.session_state.fig_clusters = fig_clusters
                st.session_state.X_train_sel = X_train_sel
                st.session_state.X_test_sel = X_test_sel
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.selected_cols = selected_cols
                st.session_state.cv_df = cv_df
                st.session_state.best_model = best_model
                st.session_state.best_name = best_name
                st.session_state.metrics = metrics
                st.session_state.test_metrics = test_metrics
                st.session_state.models_for_plot = models_for_plot
                st.session_state.fig_roc = fig_roc
                st.session_state.y_pred = y_pred
                st.session_state.fig_cm = fig_cm
                st.session_state.fig_fi = fig_fi
                st.session_state.explanations = explanations

                st.rerun()

# ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ (всегда, если есть данные) ====================
if st.session_state.analysis_completed:
    df = st.session_state.df
    all_features = st.session_state.all_features
    target_col = st.session_state.get('target_column', 'risk_flag')

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
            correlation_analysis_enhanced,
            df, all_features, target_col,
            corr_threshold=corr_threshold,
            error_msg="Ошибка при расчёте корреляций"
        )

        if corr_results:
            fig_corr = px.imshow(
                corr_results['full_matrix'],
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                title=f"Корреляционная матрица (порог сильных связей: {corr_threshold})"
            )
            st.plotly_chart(fig_corr)

            # Показываем только сильные корреляции с target
            st.write("**Топ корреляций с целевой переменной:**")
            st.dataframe(corr_results['target_correlations'].head(10))

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
    st.write(", ".join(st.session_state.selected_cols))

    # Кросс-валидация
    st.subheader("📊 Кросс-валидация")
    st.dataframe(st.session_state.cv_df)

    # Метрики на тесте
    st.subheader("📈 Метрики на тестовой выборке")
    test_metrics = st.session_state.test_metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F1-score", f"{test_metrics['f1']:.4f}")
    with col2:
        st.metric("ROC-AUC", f"{test_metrics['roc_auc']:.4f}")
    with col3:
        st.metric("Precision", f"{test_metrics['precision']:.4f}")
    with col4:
        st.metric("Recall", f"{test_metrics['recall']:.4f}")

    # ROC-кривая
    if st.session_state.fig_roc:
        st.plotly_chart(st.session_state.fig_roc)

    # Confusion Matrix
    st.plotly_chart(st.session_state.fig_cm)

    # Важность признаков
    if st.session_state.fig_fi:
        st.plotly_chart(st.session_state.fig_fi)

    st.subheader("📊 Продвинутый анализ")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Кросс-таблицы",
        "📁 История экспериментов",
        "📉 Временные ряды",
        "🔧 Обработка пропусков",
        "Конструктор композитных оценок"

    ])

    with tab1:
        st.write("### Кросс-таблицы")

        # Выбор переменных для кросс-таблицы
        all_cols = df.columns.tolist()

        # Определяем категориальные колонки (с <20 уникальными значениями)
        categorical_cols = []
        for col in all_cols:
            n_unique = df[col].nunique()
            # Для числовых колонок — только если уникальных значений мало
            if n_unique < 20:
                categorical_cols.append(col)

        # Если нет категориальных, предлагаем сгенерировать бины
        if len(categorical_cols) < 2:
            st.warning("⚠️ Недостаточно категориальных колонок для кросс-таблицы")
            st.info("Вы можете выбрать числовые колонки, они будут автоматически разбиты на группы")

            # Разрешаем любые колонки, но с предупреждением
            row_var = st.selectbox("Переменная для строк", all_cols)
            col_var = st.selectbox("Переменная для столбцов", all_cols)
            value_var = st.selectbox("Переменная для агрегации (опционально)", ["None"] + all_cols)

            use_binning = st.checkbox("Автоматически разбить на группы (для числовых колонок)", value=True)

            if st.button("Построить кросс-таблицу"):
                with st.spinner("Построение кросс-таблицы..."):
                    from ml_core.crosstab import create_crosstab

                    df_work = df.copy()

                    # Бинаризуем числовые колонки, если нужно
                    if use_binning:
                        for var in [row_var, col_var]:
                            if var in df_work.columns and df_work[var].dtype in [np.number] and df_work[
                                var].nunique() > 10:
                                # Разбиваем на квартили или децили
                                df_work[f"{var}_group"] = pd.qcut(df_work[var], q=4,
                                                                  labels=['Низкий', 'Ниже среднего', 'Выше среднего',
                                                                          'Высокий'], duplicates='drop')
                                if var == row_var:
                                    row_var = f"{var}_group"
                                else:
                                    col_var = f"{var}_group"

                    if value_var != "None":
                        result = create_crosstab(df_work, row_var, col_var, values=value_var, aggfunc='mean')
                    else:
                        result = create_crosstab(df_work, row_var, col_var)

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        if st.button("Экспорт таблицы в CSV"):
                            result['table'].to_csv("crosstab.csv", encoding="utf-8")
                            st.success("Сохранено как crosstab.csv")
                    with col_exp2:
                        if st.button("Экспорт таблицы в Excel"):
                            export_crosstab(result, filename="crosstab", format="excel")
                            st.success("Сохранено как crosstab.xlsx")

                    st.dataframe(result['table'])

                    if result['chi2_test']:
                        st.write(f"**Хи-квадрат тест:** p-value = {result['chi2_test']['p_value']:.4f}")
                        st.write(
                            f"**Статистически значимо:** {'✅ Да' if result['chi2_test']['significant'] else '❌ Нет'}")

                    if result['table'].shape[0] * result['table'].shape[1] < 1000:
                        st.plotly_chart(result['heatmap'], use_container_width=True)
                        st.plotly_chart(result['stacked_bar'], use_container_width=True)
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
                        result = create_crosstab(df, row_var, col_var, values=value_var, aggfunc='mean')
                    else:
                        result = create_crosstab(df, row_var, col_var)

                    st.dataframe(result['table'])

                    if result['chi2_test']:
                        st.write(f"**Хи-квадрат тест:** p-value = {result['chi2_test']['p_value']:.4f}")
                        st.write(
                            f"**Статистически значимо:** {'✅ Да' if result['chi2_test']['significant'] else '❌ Нет'}")

                    if result['table'].shape[0] * result['table'].shape[1] < 1000:
                        st.plotly_chart(result['heatmap'], use_container_width=True)
                        st.plotly_chart(result['stacked_bar'], use_container_width=True)
                    else:
                        st.info("Графики скрыты для производительности (слишком большая таблица)")
    with tab2:
        st.write("### История экспериментов")

        from ml_core.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()
        experiments = tracker.list_experiments()

        if not experiments.empty:
            st.dataframe(experiments)

            selected_exp = st.selectbox("Выберите эксперимент для загрузки", experiments['id'].tolist())
            if st.button("Загрузить эксперимент"):
                exp_data = tracker.load_experiment(selected_exp)
                st.json(exp_data['metrics'])
                if 'model' in exp_data:
                    st.success(f"Модель загружена: {exp_data['model_name']}")
        else:
            st.info("Нет сохраненных экспериментов")

        # Кнопка сохранения текущего эксперимента
        exp_name = st.text_input("Название эксперимента")
        if st.button("Сохранить текущий анализ") and exp_name:
            exp_data = {
                'metrics': test_metrics,
                'features': st.session_state.selected_cols,
                'model_name': st.session_state.best_name if st.session_state.best_name else 'unknown',
                'n_samples': len(st.session_state.df) if st.session_state.df is not None else 0,
                'description': exp_name
            }
            exp_id = tracker.save_experiment(exp_name, exp_data)
            st.success(f"Эксперимент сохранен: {exp_id}")
    with tab3:
        st.write("### Анализ временных рядов")

        from ml_core.timeseries import detect_negative_dynamics, analyze_student_trajectory

        # Проверяем наличие необходимых колонок
        if 'semester' not in df.columns:
            st.warning("⚠️ Для анализа временных рядов нужна колонка 'semester'")
            st.info("Вы можете добавить семестр в данные или использовать синтетическую генерацию с semester")

            # Предлагаем сгенерировать семестры
            if st.button("Сгенерировать тестовые семестры"):
                # Добавляем искусственные семестры
                df['semester'] = np.random.choice([1, 2, 3], size=len(df), p=[0.4, 0.3, 0.3])
                st.success("Добавлена колонка 'semester' с тестовыми значениями")
                st.rerun()
        else:
            # Проверяем наличие колонки для анализа
            available_metrics = [c for c in df.columns if c not in ['student_id', 'semester', 'cluster', target_col]]
            numeric_metrics = [c for c in available_metrics if df[c].dtype in [np.number]]

            if not numeric_metrics:
                st.warning("Нет числовых колонок для анализа динамики")
            else:
                value_col = st.selectbox("Показатель для анализа", numeric_metrics)

                if st.button("Выявить студентов с отрицательной динамикой"):
                    with st.spinner("Анализ..."):
                        dynamics = detect_negative_dynamics(
                            df,
                            student_id_col='student_id',
                            time_col='semester',
                            value_col=value_col,
                            threshold=-0.05
                        )

                        if 'error' in dynamics:
                            st.error(dynamics['error'])
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Проанализировано студентов", dynamics['n_students_analyzed'])
                            with col2:
                                st.metric("Студентов с отрицательной динамикой", len(dynamics['at_risk_students']))
                                st.metric("Процент риска по динамике", f"{dynamics['risk_percentage']:.1f}%")

                            if not dynamics['at_risk_students'].empty:
                                st.write("**Студенты с отрицательной динамикой:**")
                                st.dataframe(
                                    dynamics['at_risk_students'][['student_id', 'trend', 'first_value', 'last_value']])

                                # Выбор студента для детального анализа
                                student_id = st.selectbox(
                                    "Выберите студента для детального анализа",
                                    dynamics['at_risk_students']['student_id'].tolist()
                                )
                                if st.button("Показать траекторию"):
                                    trajectory = analyze_student_trajectory(
                                        df, student_id,
                                        time_col='semester',
                                        value_col=value_col
                                    )
                                    if trajectory['figure']:
                                        st.plotly_chart(trajectory['figure'])
                                    st.write(f"**Тренд:** {trajectory['trend']:.3f}")
                                    st.write(f"**Статус:** {trajectory['status']}")
                            else:
                                st.success("✅ Студентов с отрицательной динамикой не обнаружено")
                if st.button("Показать траекторию"):
                    trajectory = safe_execute(analyze_student_trajectory, df, student_id, value_col=value_col)
                    if trajectory and 'figure' in trajectory:
                        st.plotly_chart(trajectory['figure'])

                # Новый блок прогноза
                if st.button("Прогнозировать на следующие семестры"):
                    forecast = safe_execute(
                        forecast_grades,
                        df, student_id, value_col=value_col, future_semesters=2
                    )
                    if forecast:
                        st.write("**Прогноз на будущие семестры:**")
                        for sem, pred in zip(forecast['future_semesters'], forecast['predictions']):
                            st.write(f"Семестр {sem}: {pred:.2f}")
    with tab4:
        st.write("### Обработка пропусков")

        from ml_core.imputation import handle_missing_values, detect_outliers

        # Показываем текущие пропуски
        missing_before = df.isna().sum().sum()
        st.write(f"**Пропусков до обработки:** {missing_before}")
        st.dataframe(df.isna().sum().to_frame('Пропуски'))

        # Выбросы
        outliers = detect_outliers(df)
        st.write("### Выбросы в данных")
        for col, report in outliers.items():
            if report['n_outliers'] > 0:
                st.write(f"**{col}:** {report['n_outliers']} выбросов ({report['percentage']:.1f}%)")

        strategy = st.selectbox("Стратегия обработки", ["auto", "fill_median", "fill_mean", "interpolate", "drop_rows"])

        if st.button("Применить обработку"):
            df_clean, report = handle_missing_values(df, strategy=strategy)
            st.write("### Отчет об обработке")
            st.write(f"**Было:** {report['original_shape']}")
            st.write(f"**Стало:** {report['final_shape']}")
            st.write(f"**Удалено пропусков:** {report['missing_before'] - report['missing_after']}")

            for action in report['actions']:
                st.write(f"- {action['column']}: {action['message']}")

            # Обновляем df в session_state
            st.session_state.df = df_clean
            st.success("Данные обновлены! Перезапустите анализ для применения изменений.")
    with tab5:  # или создай новую вкладку "Композитные оценки"
        st.write("### Конструктор композитных оценок")

        available_features = [col for col in all_features if df[col].dtype in [np.number]]

        weights = {}
        col1, col2 = st.columns(2)
        for i, feat in enumerate(available_features[:8]):  # ограничим для удобства
            with col1 if i % 2 == 0 else col2:
                weight = st.slider(f"Вес для {feat}", -1.0, 1.0, 0.0, 0.05, key=f"w_{feat}")
                weights[feat] = weight

        score_name = st.text_input("Название новой оценки", "custom_risk_index")

        if st.button("Создать композитную оценку"):
            df_new, new_col = safe_execute(
                build_composite_score,
                df, weights, score_name,
                error_msg="Ошибка при создании композитной оценки"
            )
            if new_col in df_new.columns:
                st.success(f"Создана оценка **{new_col}**")
                st.dataframe(df_new[[new_col]].describe())
                # Добавь в all_features для дальнейшего использования
                st.session_state.all_features.append(new_col)
                st.session_state.df = df_new
    # SHAP объяснения
    st.subheader("💡 Объяснения предсказаний")
    if st.session_state.explanations:
        for exp in st.session_state.explanations[:3]:
            with st.expander(f"Студент #{exp['student_index']} - Риск: {exp['risk_probability']:.1%}"):
                st.markdown(exp['explanation'])
    else:
        st.info("SHAP объяснения не сгенерированы")

# ==================== ЭКСПОРТ (всегда внизу) ====================
st.subheader("💾 Экспорт результатов")

if st.session_state.analysis_completed:
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.X_test_sel is not None:
            predictions_df = pd.DataFrame(
                st.session_state.X_test_sel,
                columns=st.session_state.selected_cols
            )
            predictions_df['risk_pred'] = st.session_state.y_pred
            predictions_df['risk_probability'] = st.session_state.best_model.predict_proba(
                st.session_state.X_test_sel
            )[:, 1]
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Скачать предсказания (CSV)",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True,
                key='download_pred'
            )

    with col2:
        if st.session_state.explanations is not None:
            explanations_df = pd.DataFrame([
                {
                    'student_index': e['student_index'],
                    'risk_probability': e['risk_probability'],
                    'risk_level': e['risk_level'],
                    'explanation': e['explanation']
                }
                for e in st.session_state.explanations
            ])
            csv_exp = explanations_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Скачать объяснения (CSV)",
                csv_exp,
                "explanations.csv",
                "text/csv",
                use_container_width=True,
                key='download_exp'
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

        # Загрузка новых данных для проверки
        new_data_file = st.file_uploader(
            "Загрузите новые данные для проверки дрейфа",
            type=['csv'],
            key='drift_upload'
        )
        use_full_reference = st.checkbox("Использовать полный датасет как эталон", value=True)

        if new_data_file is not None:
            new_df = pd.read_csv(new_data_file)
            current_cols = st.session_state.selected_cols

            # Проверяем наличие нужных признаков
            if all(col in new_df.columns for col in current_cols):
                new_data = new_df[current_cols].fillna(new_df[current_cols].median())

                if st.button("🔍 Проверить дрейф"):
                    with st.spinner("Анализ дрейфа данных..."):

                        if use_full_reference and st.session_state.X_train_sel is not None:
                            full_data = pd.DataFrame(
                                np.vstack([st.session_state.X_train_sel, st.session_state.X_test_sel]),
                                columns=st.session_state.selected_cols
                            )
                            st.session_state.drift_detector.reference_data = full_data
                        else:
                            st.session_state.drift_detector.reference_data = st.session_state.reference_data

                        drift_report = st.session_state.drift_detector.detect_drift(new_data)
                        st.session_state.drift_report = drift_report

                        # Отображаем результаты
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Дрейфующих признаков", len(drift_report['drifted_features']))
                        with col2:
                            st.metric("Процент дрейфа", f"{drift_report['drift_percentage']:.1f}%")
                        with col3:
                            status = "🔴 Есть дрейф" if drift_report['overall_drift'] else "🟢 Нет дрейфа"
                            st.metric("Статус", status)

                        # Показываем предупреждение
                        if drift_report['overall_drift']:
                            st.warning(st.session_state.drift_detector.generate_alert_message(drift_report))
                        else:
                            st.success("✅ Данные стабильны, модель можно использовать")

                        # Детальный отчет
                        with st.expander("Детальный отчет о дрейфе"):
                            st.json(drift_report)

                        # Сохраняем отчет
                        report_path = st.session_state.drift_detector.save_report(drift_report)
                        st.info(f"Отчет сохранен: {report_path}")
            else:
                missing = set(current_cols) - set(new_df.columns)
                st.error(f"В загруженных данных отсутствуют признаки: {missing}")

    with tab2:
        st.write("### История метрик модели")

        # Показываем историю метрик из логов
        if os.path.exists('logs/model_metrics.csv'):
            history_df = pd.read_csv('logs/model_metrics.csv')
            st.dataframe(history_df)

            # График изменения F1-score
            if len(history_df) > 1:
                fig_history = px.line(
                    history_df,
                    x='timestamp',
                    y='f1_score',
                    title='Динамика качества модели'
                )
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
            help="Чем меньше значение, тем чувствительнее детектор"
        )

        if st.button("Обновить порог"):
            st.session_state.drift_detector.threshold = new_threshold
            st.success(f"Порог обновлен на {new_threshold}")

        # Кнопка для переобучения
        if st.button("🔄 Запланировать переобучение"):
            st.info("Функция переобучения будет доступна в следующей версии")
