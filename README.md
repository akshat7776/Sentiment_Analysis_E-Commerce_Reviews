# Sentiment Analysis on E-Commerce Reviews

A comprehensive **sentiment analysis pipeline** for e-commerce reviews using multiple state-of-the-art approaches: **VADER** (lexicon-based), **RoBERTa** (transformer-based), and **Rating-based** sentiment classification.

## Features

- **Multiple Sentiment Analysis Methods**:
  - **VADER**: Valence Aware Dictionary and sEntiment Reasoner (lexicon-based)
  - **RoBERTa**: Robustly Optimized BERT Pretraining Approach (transformer-based)
  - **Rating-based**: Rule-based sentiment mapping from star ratings
  
- **Comprehensive Analysis**:
  - Data preprocessing and cleaning
  - Text feature extraction and normalization
  - Sentiment distribution visualization
  - Method comparison and correlation analysis

- **Production Ready**:
  - Modular architecture with clean separation of concerns
  - CLI interface for easy execution
  - Sample data included for immediate testing
  - Comprehensive error handling and user guidance

## Folder Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── outputs/                     # Analysis results and sample data
│   ├── sample_of_data.csv      # Sample dataset for testing
│   ├── pipeline_results.csv    # Analysis results
│   ├── polarity.png            # Polarity distribution charts
│   ├── Rating Sentiment.png    # Rating-based sentiment visualization
│   └── vader_roBERTa.png       # Method comparison charts
└── sentiment_analysis/         # Main package
    ├── __init__.py             # Package initialization
    ├── sentiment-analysis-vader-roberta.ipynb  # Jupyter notebook
    └── SRC/                    # Source modules
        ├── __init__.py         # Subpackage initialization
        ├── data_preprocessing.py  # Data loading and preprocessing
        ├── Vader.py            # VADER sentiment analysis
        ├── roBERTa.py          # RoBERTa transformer analysis
        └── Pipeline.py         # Main execution pipeline
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/akshat7776/Sentiment_Analysis_E-Commerce_Reviews.git Sentiment_Analysis_E-Commerce_Reviews
cd Sentiment_Analysis_E-Commerce_Reviews

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('vader_lexicon')"
```

**Verify installation:**
```bash
# Test if all packages are installed correctly
python -c "import pandas, numpy, nltk, transformers, torch, scipy, matplotlib; print('All packages installed successfully!')"
```

### 2. Run Analysis

**Option A: Use Sample Data (No setup required)**
```bash
python -m sentiment_analysis.SRC.Pipeline
```

**Option B: Analyze Your Own Data**
```bash
python -m sentiment_analysis.SRC.Pipeline "path/to/your/reviews.csv"
```

**Option C: From Source Directory**
```bash
cd sentiment_analysis/SRC
python Pipeline.py "your_data.csv"
```

### 3. CSV Format Requirements

Your dataset should contain these columns:
- **`Review Text`**: Customer review text
- **`Rating`**: Numerical rating (1-5 stars)

```csv
Review Text,Rating
"Amazing product! Highly recommend.",5
"Poor quality, very disappointed.",1
"It's okay, nothing special.",3
```

## What You'll Get

### Output Analysis
- **Sentiment Classifications**: Three different sentiment scores for each review
- **Comparative Charts**: Visual comparison between VADER, RoBERTa, and Rating-based methods
- **Detailed Reports**: Statistics, correlations, and distribution analysis
- **CSV Export**: Complete results saved for further analysis

### Sample Output
```
Sentiment Analysis Pipeline
========================================
Using sample file: outputs/sample_of_data.csv

Running sentiment analysis...
Processed 50 reviews

Generating visualizations...
Generating report...

Sentiment Analysis Report
==============================
Total reviews: 50
VADER-RoBERTa correlation: 0.134

Rating_Sentiment distribution:
positive    80.0%
neutral     12.0%
negative     8.0%

Results saved to: processed_data.csv
Analysis complete!
```

## Methodology

### 1. VADER Sentiment Analysis
- **Type**: Lexicon and rule-based
- **Strengths**: Fast execution, optimized for social media text, handles intensifiers and punctuation
- **Output**: Compound score (-1 to +1) with individual negative/neutral/positive scores
- **Best for**: Quick analysis, social media content, informal text

### 2. RoBERTa Sentiment Analysis  
- **Type**: Transformer-based deep learning model
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Strengths**: Context-aware analysis, handles complex language patterns, state-of-the-art accuracy
- **Output**: Probability scores for negative/neutral/positive classes
- **Best for**: Nuanced analysis, formal reviews, complex sentiment detection

### 3. Rating-based Sentiment
- **Type**: Rule-based mapping from numerical ratings
- **Logic**: 
  - Ratings 1-2 → Negative sentiment
  - Rating 3 → Neutral sentiment  
  - Ratings 4-5 → Positive sentiment
- **Best for**: Baseline comparison, validation of other methods

## Visualizations

The pipeline generates comprehensive visualizations:

1. **Sentiment Distribution Pie Charts**: Compare sentiment distributions across all three methods
2. **Polarity Score Distributions**: Density plots showing score distributions
3. **Rating vs Sentiment Analysis**: Box plots and correlation analysis
4. **Method Agreement Analysis**: Correlation matrices and agreement statistics

## API Usage

### Programmatic Usage

```python
from sentiment_analysis.SRC.Pipeline import run_complete_analysis, plot_sentiment_comparison, generate_report

# Run complete analysis pipeline
df = run_complete_analysis("your_data.csv")

# Generate comparison visualizations
plot_sentiment_comparison(df)

# Get detailed statistical report
generate_report(df)

# Access individual results
print(df[['Review Text', 'Rating', 'vader_sentiment', 'roberta_sentiment', 'Rating_Sentiment']].head())
```

### Individual Module Usage

```python
from sentiment_analysis.SRC.data_preprocessing import prepare_dataset
from sentiment_analysis.SRC.Vader import run_vader_analysis
from sentiment_analysis.SRC.roBERTa import run_roberta_analysis

# Step-by-step analysis
df = prepare_dataset('reviews.csv')
df_with_vader = run_vader_analysis(df)
df_complete = run_roberta_analysis(df_with_vader)
```

## Configuration & Customization

The pipeline uses sensible defaults but can be customized:

### Data Preprocessing
- Text cleaning and normalization
- Rating validation and mapping
- Missing data handling

### Model Parameters
- VADER: Uses default lexicon with compound scoring
- RoBERTa: Cardiff NLP Twitter-optimized model
- Batch processing for efficiency

## Dataset Information

### Compatible Data Sources
- E-commerce review datasets
- Social media sentiment data
- Customer feedback surveys
- Product review platforms

### Sample Dataset
The project includes a sample dataset (`outputs/sample_of_data.csv`) with 50 real e-commerce reviews for immediate testing.

## Performance & Compatibility

### Processing Speed
- **VADER**: ~1000 reviews/second (very fast)
- **RoBERTa**: ~10-50 reviews/second (depends on hardware)
- **Combined Pipeline**: Optimized for batch processing
