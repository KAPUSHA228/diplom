import pandas as pd
import numpy as np


def extract_text_features(df: pd.DataFrame, text_column: str = "essay_text") -> pd.DataFrame:
    """Простая обработка текста эссе → числовые признаки"""
    df = df.copy()

    if text_column not in df.columns:
        return df

    df[f"{text_column}_length"] = df[text_column].astype(str).str.len()
    df[f"{text_column}_word_count"] = df[text_column].astype(str).str.split().str.len()

    # Простой маркер "сложности" (соотношение длины)
    df[f"{text_column}_complexity"] = df[f"{text_column}_length"] / (df[f"{text_column}_word_count"] + 1)

    return df