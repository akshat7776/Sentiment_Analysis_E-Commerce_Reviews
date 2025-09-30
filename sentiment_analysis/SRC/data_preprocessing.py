import pandas as pd
import numpy as np
import re

def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df[['Review Text', 'Rating']].copy()
    df_new = df_new.dropna(subset=['Review Text'])
    return df_new

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def rating_sentiment(rating):
    if rating in [1, 2]:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    elif rating in [4, 5]:
        return 'positive'
    else:
        return 'unknown'

def prepare_dataset(file_path: str) -> pd.DataFrame:
    df = load_dataset(file_path)
    df_clean = clean_data(df)
    df_clean['Rating_Sentiment'] = df_clean['Rating'].apply(rating_sentiment)
    df_clean['preprocess_text'] = df_clean['Review Text'].apply(preprocess_text)
    return df_clean
