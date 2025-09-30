import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import prepare_dataset
from Vader import run_vader_analysis
from roBERTa import run_roberta_analysis

def run_complete_analysis(file_path: str):
    df = prepare_dataset(file_path)
    df = run_vader_analysis(df)
    df = run_roberta_analysis(df)
    return df

def plot_sentiment_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (col, title) in enumerate([('Rating_Sentiment', 'Rating-based'), ('vader_sentiment', 'VADER'), ('roberta_sentiment', 'RoBERTa')]):
        if col in df.columns:
            counts = df[col].value_counts(normalize=True)
            axes[i].pie(counts.values, labels=counts.index, autopct='%.1f%%', colors=['red', 'blue', 'green'])
            axes[i].set_title(f'{title} Sentiment')
    
    plt.tight_layout()
    plt.show()

def generate_report(df: pd.DataFrame):
    print("Sentiment Analysis Report")
    print("=" * 30)
    print(f"Total reviews: {len(df)}")
    
    if 'vader_polarity' in df.columns and 'roberta_polarity' in df.columns:
        correlation = df['vader_polarity'].corr(df['roberta_polarity'])
        print(f"VADER-RoBERTa correlation: {correlation:.3f}")
    
    for col in ['Rating_Sentiment', 'vader_sentiment', 'roberta_sentiment']:
        if col in df.columns:
            print(f"\n{col} distribution:")
            print(df[col].value_counts(normalize=True).round(3))

if __name__ == "__main__":
    df = run_complete_analysis("../../Womens Clothing E-Commerce Reviews.csv")
    plot_sentiment_comparison(df)
    generate_report(df)