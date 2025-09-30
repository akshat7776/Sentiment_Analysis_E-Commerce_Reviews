# Sentiment Analysis on E-Commerce Reviews

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

A comprehensive **sentiment analysis pipeline** for e-commerce reviews using multiple state-of-the-art approaches: **VADER** (lexicon-based), **RoBERTa** (transformer-based), and **Rating-based** sentiment classification.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [What You'll Get](#what-youll-get)
- [Methodology](#methodology)
- [Visualizations](#visualizations)
- [API Usage](#api-usage)
- [Configuration & Customization](#configuration--customization)
- [Dataset Information](#dataset-information)
- [Performance & Compatibility](#performance--compatibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

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
  - Automated reporting with statistics

- **Production Ready**:
  - Modular architecture with clean separation of concerns
  - CLI interface for easy execution
  - Sample data included for immediate testing
  - Comprehensive error handling and user guidance

## Project Structure

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

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`
- Internet connection for downloading NLTK data and transformer models

## Quick Start

### 1. Installation

**Prerequisites:**
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

**Install the project:**

```bash
# Clone the repository
git clone https://github.com/akshat7776/Sentiment_Analysis_E-Commerce_Reviews.git
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

Results saved to: pipeline_results.csv
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

### Hardware Requirements
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB+ RAM, GPU for faster RoBERTa processing
- **Storage**: ~2GB for models and dependencies

### Tested Platforms
- **Operating Systems**: Windows 10/11, macOS, Linux (Ubuntu, CentOS)
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **Hardware**: CPU-only and GPU-accelerated systems

## Troubleshooting

### Common Issues

**ImportError: No module named 'transformers'**
```bash
pip install --upgrade transformers torch
```

**NLTK Data Not Found**
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

**Memory Error with RoBERTa**
- Reduce batch size in processing
- Use CPU-only mode: set `CUDA_VISIBLE_DEVICES=""`
- Close other applications to free memory

**File Not Found Error**
- Ensure CSV file path is correct
- Check file permissions
- Verify CSV has required columns: 'Review Text' and 'Rating'

### Getting Help
- Check the error message carefully
- Ensure all dependencies are installed
- Try running with sample data first
- Open an issue on GitHub with error details

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Author

**Aksha** - [akshat7776](https://github.com/akshat7776)

Feel free to reach out for questions, suggestions, or collaborations!

## Acknowledgments

- [NLTK](https://www.nltk.org/) for VADER sentiment analysis
- [Hugging Face](https://huggingface.co/) for transformer models
- [Cardiff NLP](https://github.com/cardiffnlp) for the RoBERTa sentiment model
- Women's Clothing E-Commerce Reviews dataset contributors

---

**Star this repository if you found it helpful!**