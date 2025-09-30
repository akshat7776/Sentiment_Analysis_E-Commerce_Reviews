import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def download_vader():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

class VaderAnalyzer:
    def __init__(self):
        download_vader()
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str):
        return self.analyzer.polarity_scores(text)
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column='preprocess_text'):
        df = df.copy()
        df["vader_scores"] = df[text_column].apply(self.analyze_text)
        df["vader_polarity"] = df["vader_scores"].apply(lambda x: x["compound"])
        df["vader_sentiment"] = pd.cut(df["vader_polarity"], bins=[-1.0, -0.25, 0.25, 1.0], labels=["Negative", "Neutral", "Positive"])
        scores_df = df['vader_scores'].apply(pd.Series)
        for col in ['neg', 'neu', 'pos', 'compound']:
            if col in scores_df.columns:
                df[col] = scores_df[col]
        return df

def run_vader_analysis(df: pd.DataFrame):
    analyzer = VaderAnalyzer()
    return analyzer.analyze_dataframe(df)