"""
Модуль для обнаружения дрейфа данных в ML-моделях
Сравнивает текущие данные с эталонными (на которых обучалась модель)
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
import logging
import threading
import time


class DriftMonitorThread:
    """Фоновый поток для мониторинга дрейфа"""

    def __init__(self, detector, check_interval_hours=24):
        self.detector = detector
        self.check_interval = check_interval_hours
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()

    def _run(self):
        while self.running:
            # Здесь можно добавить автоматическую загрузку новых данных
            # и проверку дрейфа
            time.sleep(self.check_interval * 3600)

    def stop(self):
        self.running = False


class DataDriftDetector:
    """
    Обнаружение дрейфа данных для моделей академического риска

    Пример использования:
        detector = DataDriftDetector(reference_data=df_train, model_metadata=metadata)
        drift_report = detector.detect_drift(df_new)
        if drift_report['overall_drift']:
            print("Обнаружен дрейф! Нужно переобучать модель.")
    """

    def __init__(self,
                 reference_data: pd.DataFrame,
                 model_metadata: Optional[Dict] = None,
                 threshold: float = 0.05,
                 model_name: str = "unknown_model"):
        """
        Args:
            reference_data: Эталонные данные (на которых обучалась модель)
            model_metadata: Метаданные модели (признаки, тип, дата обучения)
            threshold: Порог p-value для определения дрейфа (обычно 0.05)
            model_name: Имя модели для логирования
        """
        self.reference_data = reference_data
        self.model_metadata = model_metadata or {}
        self.threshold = threshold
        self.model_name = model_name

        # Определяем типы признаков
        self.numerical_features = self.model_metadata.get('numerical_features',
                                                          reference_data.select_dtypes(
                                                              include=[np.number]).columns.tolist())
        self.categorical_features = self.model_metadata.get('categorical_features',
                                                            reference_data.select_dtypes(
                                                                include=['object', 'category']).columns.tolist())

        # Вычисляем статистики по эталонным данным
        self.ref_stats = self._compute_statistics(reference_data)

        # Настройка логирования
        self.logger = logging.getLogger('drift_detector')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Вычисление статистик по данным"""
        df = df.copy()

        stats = {
            'numerical': {},
            'categorical': {},
            'general': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'timestamp': datetime.now().isoformat()
            }
        }

        # Статистики для числовых признаков
        for feat in self.numerical_features:
            if feat in df.columns:
                stats['numerical'][feat] = {
                    'mean': float(df[feat].mean()),
                    'std': float(df[feat].std()),
                    'min': float(df[feat].min()),
                    'max': float(df[feat].max()),
                    'median': float(df[feat].median()),
                    'q25': float(df[feat].quantile(0.25)),
                    'q75': float(df[feat].quantile(0.75)),
                    'missing_pct': float(df[feat].isna().mean() * 100)
                }

        # Статистики для категориальных признаков
        for feat in self.categorical_features:
            if feat in df.columns:
                value_counts = df[feat].value_counts(normalize=True)
                stats['categorical'][feat] = {
                    'unique_values': int(df[feat].nunique()),
                    'value_counts': {str(k): float(v) for k, v in value_counts.head(10).items()},
                    'missing_pct': float(df[feat].isna().mean() * 100),
                    'mode': str(df[feat].mode()[0]) if not df[feat].mode().empty else None
                }

        return stats

    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Обнаружение дрейфа в текущих данных

        Args:
            current_data: Новые данные для проверки

        Returns:
            Dict с отчетом о дрейфе
        """

        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'overall_drift': False,
            'drift_percentage': 0.0,
            'drifted_features': [],
            'feature_reports': {},
            'recommendations': [],
            'data_quality': {}
        }

        # Проверяем качество данных
        drift_report['data_quality'] = self._check_data_quality(current_data)

        # Проверка числовых признаков (KS-тест)
        for feat in self.numerical_features:
            if feat not in current_data.columns:
                drift_report['feature_reports'][feat] = {
                    'error': 'Feature missing in current data',
                    'drifted': True
                }
                drift_report['drifted_features'].append(feat)
                continue

            ref_values = self.reference_data[feat].dropna()
            curr_values = current_data[feat].dropna()

            if len(ref_values) > 0 and len(curr_values) > 0:
                # Тест Колмогорова-Смирнова
                ks_stat, p_value = ks_2samp(ref_values, curr_values)

                # Статистические различия
                ref_mean = ref_values.mean()
                curr_mean = curr_values.mean()
                mean_diff_pct = abs((curr_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0

                drifted = p_value < self.threshold

                drift_report['feature_reports'][feat] = {
                    'type': 'numerical',
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drifted': drifted,
                    'ref_mean': float(ref_mean),
                    'curr_mean': float(curr_mean),
                    'mean_diff_pct': float(mean_diff_pct),
                    'ref_std': float(ref_values.std()),
                    'curr_std': float(curr_values.std()),
                }

                if drifted:
                    drift_report['drifted_features'].append(feat)

        # Проверка категориальных признаков (Хи-квадрат)
        for feat in self.categorical_features:
            if feat not in current_data.columns:
                continue

            # Создаем таблицу сопряженности
            ref_counts = self.reference_data[feat].value_counts()
            curr_counts = current_data[feat].value_counts()

            # Выравниваем категории
            all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]

            if len(ref_aligned) > 1 and len(curr_aligned) > 1 and sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                from scipy.stats import chi2_contingency
                contingency = np.array([ref_aligned, curr_aligned])
                chi2, p_value, dof, expected = chi2_contingency(contingency)

                drifted = p_value < self.threshold

                drift_report['feature_reports'][feat] = {
                    'type': 'categorical',
                    'chi2_statistic': float(chi2),
                    'p_value': float(p_value),
                    'drifted': drifted,
                    'ref_distribution': {str(k): float(v) for k, v in ref_counts.to_dict().items()},
                    'curr_distribution': {str(k): float(v) for k, v in curr_counts.to_dict().items()},
                }

                if drifted:
                    drift_report['drifted_features'].append(feat)

        # Общий вердикт
        total_features = len(self.numerical_features) + len(self.categorical_features)
        if total_features > 0:
            drift_report['drift_percentage'] = len(drift_report['drifted_features']) / total_features * 100
            drift_report['overall_drift'] = len(drift_report['drifted_features']) > 0

            # Генерируем рекомендации
            if drift_report['overall_drift']:
                if drift_report['drift_percentage'] > 30:
                    drift_report['recommendations'].append(
                        "🚨 КРИТИЧЕСКИЙ ДРЕЙФ: требуется немедленное переобучение модели"
                    )
                elif drift_report['drift_percentage'] > 10:
                    drift_report['recommendations'].append(
                        "⚠️ УМЕРЕННЫЙ ДРЕЙФ: рекомендуется переобучить модель в ближайшее время"
                    )
                else:
                    drift_report['recommendations'].append(
                        "ℹ️ НЕБОЛЬШОЙ ДРЕЙФ: следите за качеством модели"
                    )

                # Добавляем конкретные признаки
                top_drifted = drift_report['drifted_features'][:5]
                drift_report['recommendations'].append(
                    f"Признаки с дрейфом: {', '.join(top_drifted)}"
                )
            else:
                drift_report['recommendations'].append(
                    "✅ Дрейф не обнаружен. Модель можно использовать дальше."
                )

        return drift_report

    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Проверка качества новых данных"""
        df = df.copy()
        quality_report = {}

        # Проверка на пропуски
        missing_data = df.isna().sum()
        missing_pct = (missing_data / len(df) * 100).to_dict()
        quality_report['missing_values'] = {str(k): float(v) for k, v in missing_pct.items() if v > 0}

        # Проверка на выбросы (для числовых признаков)
        outliers = {}
        for feat in self.numerical_features:
            if feat in df.columns:
                q1 = df[feat].quantile(0.25)
                q3 = df[feat].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_mask = (df[feat] < lower_bound) | (df[feat] > upper_bound)
                outliers[feat] = float(outlier_mask.mean() * 100)

        quality_report['outliers_percentage'] = outliers

        # Проверка на новые категории
        new_categories = {}
        for feat in self.categorical_features:
            if feat in df.columns and feat in self.ref_stats['categorical']:
                ref_cats = set(self.reference_data[feat].unique())
                curr_cats = set(df[feat].unique())
                new_cats = curr_cats - ref_cats
                if new_cats:
                    new_categories[feat] = list(new_cats)[:5]

        quality_report['new_categories'] = new_categories

        return quality_report

    def generate_alert_message(self, drift_report: Dict) -> str:
        """Генерирует человеко-читаемое сообщение для алерта"""
        if not drift_report['overall_drift']:
            return f"✅ Модель {self.model_name}: дрейф не обнаружен"

        msg = f"⚠️ ДРЕЙФ ДАННЫХ в модели {self.model_name}\n"
        msg += f"Дрейфующих признаков: {len(drift_report['drifted_features'])} ({drift_report['drift_percentage']:.1f}%)\n"

        if drift_report['recommendations']:
            msg += "\nРекомендации:\n"
            for rec in drift_report['recommendations'][:3]:
                msg += f"• {rec}\n"

        return msg

    def save_report(self, drift_report: Dict, report_dir: str = 'logs/drift_reports'):
        """Сохраняет отчет о дрейфе в файл"""
        os.makedirs(report_dir, exist_ok=True)

        filename = f"{report_dir}/drift_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            # Конвертируем numpy типы в Python типы для JSON
            json.dump(drift_report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Drift report saved to {filename}")
        return filename


class DriftMonitorScheduler:
    """
    Планировщик для регулярной проверки дрейфа моделей
    """

    def __init__(self, check_interval_hours: int = 24):
        self.check_interval = check_interval_hours
        self.detectors = {}
        self.last_checks = {}

    def register_model(self, model_name: str, detector: DataDriftDetector):
        """Регистрирует модель для мониторинга"""
        self.detectors[model_name] = detector
        self.last_checks[model_name] = None

    def check_all_models(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """Проверяет все зарегистрированные модели"""
        results = {}
        for name, detector in self.detectors.items():
            try:
                results[name] = detector.detect_drift(current_data)
                self.last_checks[name] = datetime.now()
            except Exception as e:
                results[name] = {'error': str(e)}

        return results

    def get_models_needing_retraining(self, drift_threshold: float = 10.0) -> List[str]:
        """Возвращает список моделей, которым нужно переобучение"""
        needing = []
        for name, detector in self.detectors.items():
            if hasattr(detector, 'last_report') and detector.last_report:
                if detector.last_report.get('drift_percentage', 0) > drift_threshold:
                    needing.append(name)
        return needing

def generate_recommendations(self, drift_report):
    recs = []
    if drift_report['overall_drift']:
        recs.append("Рекомендуется переобучить модель на новых данных")
        if len(drift_report['drifted_features']) > 3:
            recs.append("Рассмотреть добавление/удаление признаков")
        recs.append("Проверить качество новых данных (пропуски, выбросы)")
    else:
        recs.append("Данные стабильны — модель можно использовать")
    return recs