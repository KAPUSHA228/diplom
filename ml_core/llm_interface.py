"""
Интерфейс к LLM для интерпретации результатов и анализа текстов
"""
import json
import os
import pandas as pd

import requests
import streamlit as st
from typing import List, Dict, Any


class LLMInterface:
    """
    Класс для работы с LLM API.
    Поддерживает YandexGPT, GigaChat, OpenAI (опционально).
    """

    def __init__(self, provider='yandex', api_key=None, folder_id=None):
        self.provider = provider
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.folder_id = folder_id or os.getenv('YANDEX_FOLDER_ID')

    def _call_yandex_gpt(self, prompt: str) -> str:
        """Вызов YandexGPT API"""
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": 2000
            },
            "messages": [
                {"role": "system",
                 "text": "Ты — эксперт по социологическим данным. Отвечай на русском языке, четко и структурированно."},
                {"role": "user", "text": prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['result']['alternatives'][0]['message']['text']
        else:
            return f"Ошибка LLM: {response.status_code}"

    def _call_gigachat(self, prompt: str) -> str:
        """Вызов GigaChat API (Сбер)"""
        # TODO: реализовать
        return "GigaChat пока не подключен"

    def complete(self, prompt: str) -> str:
        """Основной метод для вызова LLM"""
        if self.provider == 'yandex':
            return self._call_yandex_gpt(prompt)
        elif self.provider == 'gigachat':
            return self._call_gigachat(prompt)
        else:
            return "LLM не настроен"

    def interpret_clusters(self, cluster_profiles: pd.DataFrame, n_clusters: int) -> str:
        """
        Интерпретация кластеров.
        cluster_profiles: DataFrame со средними значениями по кластерам
        """
        # Формируем промпт
        prompt = f"""
        Проанализируй следующие данные о {n_clusters} кластерах студентов.
        Для каждого кластера даны средние значения по признакам.

        Данные:
        {cluster_profiles.to_string()}

        Задача:
        1. Дай каждому кластеру короткое название (например, "Творческие активисты")
        2. Опиши ключевые характеристики каждого кластера
        3. Сравни кластеры между собой
        4. Предложи возможные рекомендации для работы с каждым типом студентов

        Ответ должен быть на русском языке, четким и структурированным.
        """

        return self.complete(prompt)

    def analyze_text_responses(self, responses: List[str], question_text: str) -> str:
        """
        Анализ текстовых ответов на открытые вопросы.
        """
        # Ограничиваем количество ответов для LLM
        sample = responses[:50] if len(responses) > 50 else responses

        prompt = f"""
        Проанализируй ответы студентов на вопрос:
        "{question_text}"

        Примеры ответов (первые {len(sample)} из {len(responses)}):
        {chr(10).join([f"- {r}" for r in sample[:20]])}

        Задача:
        1. Выдели основные категории ответов
        2. Определи наиболее частые паттерны
        3. Напиши краткое резюме

        Ответ должен быть на русском языке.
        """

        return self.complete(prompt)

    def generate_report(self, metrics: Dict, correlations: Dict,
                        feature_importance: Dict, clusters_desc: str) -> str:
        """
        Генерация итогового отчета по результатам анализа.
        """
        prompt = f"""
        На основе проведенного анализа данных составь исследовательский отчет.

        Метрики моделей:
        {json.dumps(metrics, ensure_ascii=False, indent=2)}

        Топ-5 корреляций с целевой переменной:
        {json.dumps(correlations, ensure_ascii=False, indent=2)}

        Важность признаков:
        {json.dumps(feature_importance, ensure_ascii=False, indent=2)}

        Кластеры студентов:
        {clusters_desc}

        Задача:
        1. Сформулируй основные выводы
        2. Выдели ключевые факторы, влияющие на целевую переменную
        3. Предложи рекомендации для дальнейших исследований

        Ответ должен быть на русском языке.
        """

        return self.complete(prompt)

    def answer_question(self, question: str, data_summary: str) -> str:
        """
        Ответ на вопрос исследователя о данных.
        """
        prompt = f"""
        Ты — эксперт по анализу социологических данных.

        Данные содержат информацию о студентах: 
        - социально-демографические характеристики
        - ценности (шкала Шварца)
        - креативность (тест Вильямса)
        - личностные качества
        - интересы и навыки
        - цифровую активность
        - участие в мероприятиях
        - успеваемость
        - трудовые намерения

        Краткая сводка:
        {data_summary}

        Вопрос исследователя: {question}

        Дай четкий, аргументированный ответ на основе данных.
        """

        return self.complete(prompt)