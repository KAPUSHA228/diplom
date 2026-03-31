"""
Сохранение и загрузка результатов экспериментов
"""
import json
import os
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ExperimentTracker:
    """
    Сохранение результатов экспериментов для возврата к ним позже.
    """

    def __init__(self, storage_dir='experiments'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _generate_id(self, name: str) -> str:
        """Генерация уникального ID эксперимента"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:6]
        return f"{name}_{timestamp}_{hash_part}"

    def save_experiment(self, name: str, data: Dict[str, Any]) -> str:
        """
        Сохраняет эксперимент.

        Parameters:
        -----------
        name : str
            Название эксперимента
        data : dict
            Данные эксперимента (модель, метрики, объяснения, графики)

        Returns:
        --------
        experiment_id : str
        """
        exp_id = self._generate_id(name)
        exp_dir = os.path.join(self.storage_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        # Сохраняем метаданные
        metadata = {
            'id': exp_id,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'metrics': data.get('metrics', {}),
            'features': data.get('features', []),
            'model_name': data.get('model_name', 'unknown'),
            'n_samples': data.get('n_samples', 0),
            'n_features': data.get('n_features', 0),
            'description': data.get('description', '')
        }

        with open(os.path.join(exp_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Сохраняем модель (если есть)
        if 'model' in data and data['model'] is not None:
            with open(os.path.join(exp_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(data['model'], f)

        # Сохраняем объяснения (если есть)
        if 'explanations' in data and data['explanations']:
            if isinstance(data['explanations'], list):
                pd.DataFrame(data['explanations']).to_csv(
                    os.path.join(exp_dir, 'explanations.csv'),
                    index=False,
                    encoding='utf-8'
                )

        # Сохраняем предсказания (если есть)
        if 'predictions' in data and data['predictions'] is not None:
            data['predictions'].to_csv(
                os.path.join(exp_dir, 'predictions.csv'),
                index=False,
                encoding='utf-8'
            )

        # Сохраняем параметры эксперимента
        if 'params' in data:
            with open(os.path.join(exp_dir, 'params.json'), 'w', encoding='utf-8') as f:
                json.dump(data['params'], f, ensure_ascii=False, indent=2)

        return exp_id

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Загружает сохраненный эксперимент.
        """
        exp_dir = os.path.join(self.storage_dir, experiment_id)

        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        # Загружаем метаданные
        with open(os.path.join(exp_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        result = metadata

        # Загружаем модель
        model_path = os.path.join(exp_dir, 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                result['model'] = pickle.load(f)

        # Загружаем объяснения
        explanations_path = os.path.join(exp_dir, 'explanations.csv')
        if os.path.exists(explanations_path):
            result['explanations'] = pd.read_csv(explanations_path)

        # Загружаем предсказания
        predictions_path = os.path.join(exp_dir, 'predictions.csv')
        if os.path.exists(predictions_path):
            result['predictions'] = pd.read_csv(predictions_path)

        # Загружаем параметры
        params_path = os.path.join(exp_dir, 'params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                result['params'] = json.load(f)

        return result

    def list_experiments(self, limit: int = 20) -> pd.DataFrame:
        """
        Список всех сохраненных экспериментов.
        """
        experiments = []

        for exp_id in os.listdir(self.storage_dir):
            exp_dir = os.path.join(self.storage_dir, exp_id)
            if os.path.isdir(exp_dir):
                meta_path = os.path.join(exp_dir, 'metadata.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    experiments.append(meta)

        if not experiments:
            return pd.DataFrame()

        df = pd.DataFrame(experiments)
        df = df.sort_values('timestamp', ascending=False).head(limit)

        return df[['id', 'name', 'timestamp', 'model_name', 'metrics']]

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Удаляет эксперимент.
        """
        import shutil
        exp_dir = os.path.join(self.storage_dir, experiment_id)

        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            return True
        return False
