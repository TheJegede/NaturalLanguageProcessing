# Natural Language Processing

A comprehensive collection of Natural Language Processing (NLP) projects and assignments covering fundamental to advanced techniques in text analysis, sentiment analysis, named entity recognition, sequence modeling, and spam detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Technologies & Libraries](#technologies--libraries)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Project Details](#project-details)
- [Contributing](#contributing)

## 🔍 Overview

This repository contains multiple Jupyter notebooks demonstrating various NLP techniques and machine learning approaches for text processing and analysis. The projects range from basic text preprocessing to advanced deep learning models for sequence prediction and classification tasks.

## 📂 Projects

### 1. **Named Entity Recognition (NER)**
- **File**: `Named Entity Recognition. ipynb`
- **Description**: Implementation of NER techniques to extract entities from text data
- **Key Tasks**:
  - Part-of-Speech (POS) tagging
  - Noun phrase chunking using regex patterns
  - Named entity extraction using NLTK's chunker
- **Dataset**: NER. csv (47,959 sentences)

### 2. **Sentiment Analysis**
- **File**: `Sentiment Analysis.ipynb`
- **Description**: Sentiment classification of product reviews using VADER sentiment analyzer
- **Key Tasks**: 
  - Text preprocessing (tokenization, lemmatization)
  - Sentiment scoring (positive, negative, neutral, compound)
  - Sentiment visualization and statistical analysis
- **Dataset**: Reviews.csv (Amazon product reviews)

### 3. **Spam Detection**
- **File**: `Spam_Detection.ipynb`
- **Description**: Email spam classification using machine learning models
- **Key Tasks**:
  - Email text preprocessing and cleaning
  - Feature extraction (TF-IDF, Count Vectorization)
  - Classification using Naive Bayes, Logistic Regression, and Random Forest
  - Model evaluation and performance comparison
- **Dataset**: Email-spam-10k (10,899 emails)

### 4. **Sequence Modeling**
- **File**: `Sequence modelling.ipynb`
- **Description**: Deep learning models for sentiment prediction on Amazon reviews
- **Key Tasks**: 
  - Text tokenization and padding
  - SimpleRNN implementation
  - Gated Recurrent Unit (GRU) implementation
  - Long Short-Term Memory (LSTM) implementation
  - Model comparison and performance evaluation
- **Dataset**:  AmazonReviews.csv (25,000 reviews)

### 5. **Topic Modeling**
- **File**:  `Topic Modeling.ipynb`
- **Description**: Discovering hidden topics in news headlines
- **Key Tasks**:
  - Explicit Semantic Analysis (ESA)
  - Agglomerative Clustering with dendrogram visualization
  - Latent Dirichlet Allocation (LDA) for topic discovery
  - Non-Negative Matrix Factorization (NMF)
- **Dataset**: abcnews-date-text. csv (ABC News headlines)

### 6. **Text Preprocessing and Cleaning**
- **File**:  `Text Preprocessing and Cleaning.ipynb`
- **Description**: Comprehensive text preprocessing pipeline
- **Techniques**:  Tokenization, stopword removal, lemmatization, stemming

### 7. **Text Representation**
- **File**: `Text Representation.ipynb`
- **Description**: Various text vectorization techniques
- **Methods**:  Bag of Words, TF-IDF, Word Embeddings

## 🛠 Technologies & Libraries

### Core Libraries
- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** & **Seaborn** - Data visualization

### NLP & Machine Learning
- **NLTK** - Natural Language Toolkit
  - Tokenization
  - POS tagging
  - Named Entity Recognition
  - Sentiment Analysis (VADER)
  - WordNet Lemmatization
- **scikit-learn** - Machine learning algorithms
  - CountVectorizer, TfidfVectorizer
  - Naive Bayes, Logistic Regression, Random Forest
  - Model evaluation metrics
  - Clustering algorithms

### Deep Learning
- **TensorFlow/Keras** - Deep learning framework
  - Sequential models
  - Embedding layers
  - SimpleRNN, LSTM, GRU layers
  - Dense layers

## ✨ Key Features

### Text Preprocessing
- Lowercasing and tokenization
- Stopword removal
- Lemmatization using WordNetLemmatizer
- Special character and punctuation removal
- Regular expression-based cleaning

### Feature Engineering
- Bag of Words representation
- TF-IDF vectorization
- Word embeddings
- Sequence padding for neural networks

### Machine Learning Models
- **Traditional ML**: Naive Bayes, Logistic Regression, Random Forest
- **Deep Learning**:  SimpleRNN, GRU, LSTM
- Model evaluation using accuracy, loss, confusion matrix, and classification reports

### Advanced NLP Techniques
- Named Entity Recognition (Person, Organization, Location)
- Sentiment intensity analysis
- Topic modeling and document clustering
- Sequence-to-sequence modeling

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn
pip install nltk scikit-learn
pip install tensorflow keras
```

### NLTK Data Downloads

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

### Running the Notebooks

1. Clone the repository:
```bash
git clone https://github.com/TheJegede/NaturalLanguageProcessing.git
cd NaturalLanguageProcessing
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open any notebook and run the cells sequentially

## 📊 Project Details

### Spam Detection Deep Learning Approach
Includes a presentation on **"Improving Email Spam Detection:  A Deep Learning Approach with GloVe Embeddings and LSTM"** demonstrating state-of-the-art spam classification techniques.

### Data Preprocessing Pipeline
All projects follow a consistent preprocessing approach:
1. Loading data from CSV files
2. Text cleaning and normalization
3. Tokenization
4. Stop word removal
5. Lemmatization
6. Feature extraction
7. Model training and evaluation

### Model Performance Tracking
Each notebook includes:
- Training and validation metrics
- Visualization of model performance
- Comparative analysis between different approaches
- Hyperparameter tuning considerations

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the code or add new NLP techniques:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**TheJegede**

- GitHub: [@TheJegede](https://github.com/TheJegede)

## 📧 Contact

For questions or feedback, please open an issue in the repository. 

---

⭐ **Star this repository if you find it helpful!**
