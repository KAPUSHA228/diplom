"""
Система анализа социологических данных
Гибридный подход: ML + LLM
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Анализ социологических данных",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Анализ социологических данных")
st.markdown("Гибридный подход: ML для структурированных данных, LLM для интерпретации")

# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки")

    # Загрузка данных
    st.subheader("📁 Загрузка данных")

    data_type = st.radio(
        "Тип данных",
        ["Синтетические данные", "CSV файл", "Excel файл (опросы)"]
    )

    uploaded_file = None

    if data_type == "CSV файл":
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
    elif data_type == "Excel файл (опросы)":
        uploaded_file = st.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])

    st.markdown("---")

    # Настройки анализа
    st.subheader("🔧 Параметры анализа")
    n_clusters = st.slider("Количество кластеров", 2, 8, 4)

    st.markdown("---")

    # Кнопка запуска
    run_analysis = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

# ============================================================
# ОСНОВНАЯ ЛОГИКА
# ============================================================

if run_analysis:

    # ----- ШАГ 1: ЗАГРУЗКА ДАННЫХ -----
    with st.spinner("Загрузка данных..."):

        if data_type == "Синтетические данные":
            # Генерируем простые синтетические данные
            np.random.seed(42)
            n = 500
            df = pd.DataFrame({
                'user_id': range(n),
                'age': np.random.randint(18, 25, n),
                'gender': np.random.choice(['М', 'Ж'], n),
                'creativity': np.random.randint(40, 160, n),
                'stress': np.random.uniform(1, 10, n),
                'motivation': np.random.uniform(1, 10, n),
                'grades': np.random.choice(['отлично', 'хорошо', 'удовлетворительно'], n, p=[0.2, 0.5, 0.3])
            })
            df['target'] = (df['grades'] == 'удовлетворительно').astype(int)
            st.success(f"✅ Сгенерировано {len(df)} записей")

        elif data_type == "CSV файл" and uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Загружено {len(df)} записей, {len(df.columns)} колонок")

        elif data_type == "Excel файл (опросы)" and uploaded_file:
            from ml_core.loader import detect_sheet_type, preprocess_sheet

            # Сохраняем временно
            with open('temp.xlsx', 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Получаем список листов
            import pandas as pd

            xl = pd.ExcelFile('temp.xlsx')
            sheet_names = xl.sheet_names

            st.write(f"📑 Найдено листов: {len(sheet_names)}")

            # Выбор листа
            selected_sheet = st.selectbox("Выберите лист для анализа", sheet_names)

            # Загружаем выбранный лист
            df_raw = pd.read_excel('temp.xlsx', sheet_name=selected_sheet)
            st.write(f"📄 Лист '{selected_sheet}': {df_raw.shape}")

            # Определяем тип листа
            sheet_type = detect_sheet_type(selected_sheet)
            st.info(f"🔍 Тип листа: {sheet_type}")

            # Предобработка в зависимости от типа
            df, info = preprocess_sheet(df_raw, sheet_type)
            st.success(f"✅ {info}")

        else:
            st.error("❌ Пожалуйста, загрузите файл")
            st.stop()

    # ----- ШАГ 2: РАЗВЕДОЧНЫЙ АНАЛИЗ -----
    st.subheader("📊 Разведочный анализ")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Количество записей", len(df))
    with col2:
        st.metric("Количество признаков", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Числовые признаки", len(numeric_cols))
    with col4:
        if 'target' in df.columns:
            risk_pct = df['target'].mean() * 100
            st.metric("Доля целевого класса", f"{risk_pct:.1f}%")

    # Показываем первые строки
    with st.expander("👁️ Просмотр данных"):
        st.dataframe(df.head(10))

    # ----- ШАГ 3: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ -----
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        st.subheader("🔗 Корреляционный анализ")

        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Корреляционная матрица"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ----- ШАГ 4: КЛАСТЕРИЗАЦИЯ -----
    if len(numeric_df.columns) >= 2:
        st.subheader("🎯 Кластеризация")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Берем числовые колонки
        X = numeric_df.fillna(numeric_df.median())

        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df['cluster'] = labels

        # Профили кластеров
        cluster_profiles = df.groupby('cluster')[numeric_df.columns].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Профили кластеров:**")
            st.dataframe(cluster_profiles)

        with col2:
            # PCA для визуализации
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            fig_clusters = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=labels,
                title=f"Кластеризация (PCA: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})"
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

    # ----- ШАГ 5: ML МОДЕЛЬ (если есть целевая переменная) -----
    if 'target' in df.columns:
        st.subheader("🤖 Машинное обучение")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        # Подготовка данных
        X = df[numeric_df.columns].fillna(df[numeric_df.columns].median())
        y = df['target']

        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Оценка
        y_pred = model.predict(X_test)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        with col2:
            st.metric("F1-score", f"{f1_score(y_test, y_pred):.3f}")
        with col3:
            st.metric("Важных признаков", len(numeric_df.columns))

        # Важность признаков
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': numeric_df.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            fig_imp = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Важность признаков'
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ----- ШАГ 6: LLM ИНТЕРПРЕТАЦИЯ -----
    st.markdown("---")
    st.subheader("🤖 AI-интерпретация")

    from ml_core.llm_interface import LLMInterface

    llm = LLMInterface(provider='yandex')

    # Кнопка для интерпретации кластеров
    if 'cluster_profiles' in locals():
        if st.button("📊 Интерпретировать кластеры"):
            with st.spinner("AI анализирует..."):
                # Формируем краткое описание
                profile_text = cluster_profiles.round(2).to_string()
                prompt = f"""
                Проанализируй следующие данные о {n_clusters} кластерах студентов.

                Данные (средние значения по кластерам):
                {profile_text}

                Задача:
                1. Дай каждому кластеру короткое название
                2. Опиши ключевые характеристики каждого кластера
                3. Сравни кластеры между собой

                Ответ должен быть на русском языке.
                """

                response = llm.complete(prompt)
                st.markdown(response)

    # Кнопка для общего отчета
    if st.button("📝 Сгенерировать отчет"):
        with st.spinner("Генерация отчета..."):
            # Собираем информацию
            info = f"""
            Данные: {len(df)} записей, {len(df.columns)} колонок
            Числовых признаков: {len(numeric_df.columns)}
            Кластеров: {n_clusters}
            """
            if 'target' in df.columns:
                info += f"\nЦелевая переменная: распределение {df['target'].value_counts().to_dict()}"

            prompt = f"""
            На основе анализа данных составь краткий исследовательский отчет.

            Информация о данных:
            {info}

            Выводы:
            - Какие основные паттерны видны в данных?
            - Какие неожиданные связи обнаружились?
            - Что стоит исследовать дальше?

            Ответ должен быть на русском языке.
            """

            response = llm.complete(prompt)
            st.markdown(response)

    # ----- ШАГ 7: ЭКСПОРТ -----
    st.markdown("---")
    st.subheader("💾 Экспорт")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Скачать данные (CSV)",
        csv,
        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

else:
    st.info("👈 Выберите тип данных и нажмите 'Запустить анализ'")

    st.markdown("""
    ### Возможности системы:

    - **Загрузка данных** — CSV, Excel (опросы), синтетические данные
    - **Разведочный анализ** — статистика, корреляции
    - **Кластеризация** — группировка студентов по схожим признакам
    - **ML-модели** — Random Forest для предсказания
    - **AI-интерпретация** — LLM объясняет кластеры и генерирует отчеты

    ### Для работы с Excel-опросниками:

    Система автоматически определяет тип листа:
    - **Вильямс** — тест креативности
    - **шварц** — ценности
    - **соц1-15** — социальные опросы
    """)