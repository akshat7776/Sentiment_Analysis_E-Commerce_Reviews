import pandas as pd
import matplotlib.pyplot as plt

try:
    from .data_preprocessing import prepare_dataset
    from .Vader import run_vader_analysis
    from .roBERTa import run_roberta_analysis
except ImportError:
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
    import os
    import sys
    
    print("Sentiment Analysis Pipeline")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Using provided file: {csv_file}")
    else:
        possible_files = [
            "../../outputs/sample_of_data.csv",
            "../outputs/sample_of_data.csv", 
            "outputs/sample_of_data.csv",
            "sample_of_data.csv"
        ]
        
        csv_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                csv_file = file_path
                break
        
        if csv_file:
            print(f"Using sample file: {csv_file}")
        else:
            print("No CSV file found!")
            print("Usage:")
            print("   python Pipeline.py 'your_file.csv'")
            print("   Or place 'sample_of_data.csv' in current directory")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        sys.exit(1)
    
    try:
        print("\nRunning sentiment analysis...")
        df = run_complete_analysis(csv_file)
        print(f"Processed {len(df)} reviews")
        
        print("\nGenerating visualizations...")
        plot_sentiment_comparison(df)
        
        print("\nGenerating report...")
        generate_report(df)
        
        output_file = "processed_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print("Analysis complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure your CSV has 'Review Text' and 'Rating' columns")
