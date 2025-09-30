import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

class RobertaAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def analyze_text(self, text: str):
        encoded_input = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = softmax(output[0][0].detach().numpy())
        return {"neg": scores[0], "neu": scores[1], "pos": scores[2]}
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column='preprocess_text'):
        df = df.copy()
        df["roberta_scores"] = df[text_column].apply(self.analyze_text)
        scores_df = df['roberta_scores'].apply(pd.Series)
        for col in ['neg', 'neu', 'pos']:
            df[f'roberta_{col}'] = scores_df[col]
        
        polarity_weights = torch.tensor([-1, 0, 1], dtype=torch.float32)
        probs = torch.tensor(df[["roberta_neg", "roberta_neu", "roberta_pos"]].values, dtype=torch.float32)
        polarity = probs @ polarity_weights
        df["roberta_polarity"] = torch.tanh(polarity).detach().numpy()
        df["roberta_sentiment"] = pd.cut(df["roberta_polarity"], bins=[-1.0, -0.25, 0.25, 1.0], labels=["Negative", "Neutral", "Positive"])
        return df

def run_roberta_analysis(df: pd.DataFrame):
    analyzer = RobertaAnalyzer()
    return analyzer.analyze_dataframe(df)