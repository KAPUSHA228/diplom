"""
Модуль для анализа данных (кластеризация, корреляция)
ВАША ЗОНА ОТВЕТСТВЕННОСТИ
"""

import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

# ---- Корреляционный анализ ----
def correlation_analysis(df, feature_cols, target_col, output_prefix="corr"):
    """
    Строит корреляционную матрицу для выбранных признаков и целевой переменной.
    Матрица сохраняется в CSV, при наличии seaborn дополнительно сохраняется heatmap в PNG.
    """
    cols = feature_cols + [target_col]
    corr = df[cols].corr()
    corr.to_csv(f"{output_prefix}_matrix.csv", encoding="utf-8", index=True)

    if sns is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_heatmap.png", dpi=200)
        plt.close()

    return corr



def cluster_students(df, n_clusters=3, feature_cols=None):
    """
    Кластеризация студентов
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[feature_cols].fillna(df[feature_cols].median())

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, kmeans, scaler


def analyze_cluster_profiles(df, features, cluster_col='cluster'):
    """
    Анализ профилей кластеров
    """
    profiles = df.groupby(cluster_col)[features].mean()
    profiles['size'] = df.groupby(cluster_col).size()
    profiles['size_pct'] = profiles['size'] / len(df) * 100

    return profiles


def plot_clusters_pca(df, labels, features):
    """
    Визуализация кластеров с помощью PCA
    """
    X = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=labels,
        title=f"Кластеризация (PCA: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})",
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Кластер'}
    )

    return fig


def plot_clusters_2d(X, cluster_labels, output_path="clusters_2d.png"):
    """
    Визуализация кластеров в 2D пространстве (PCA).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Кластер')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Кластеризация студентов (PCA проекция)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return pca
