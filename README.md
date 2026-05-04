# Toxic Comment Classification

A machine learning project for classifying toxic comments using Natural Language Processing (NLP) techniques and machine learning algorithms.

## Project Overview

This project implements an automated system to classify comments into different toxicity categories. It uses text preprocessing techniques and a Logistic Regression model with TF-IDF vectorization to identify and categorize toxic comments.

## Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the comment dataset
- **Text Preprocessing**: 
  - Stopword removal
  - Text cleaning and normalization
  - Stemming
  - Special character handling
- **Machine Learning Model**: Logistic Regression with One-vs-Rest classification
- **Multi-label Classification**: Supports multiple toxicity categories
- **Real-time Predictions**: Classify unseen comments on demand

## Dataset

- Source: Comment text data with multiple toxicity labels
- Format: CSV file containing comment text and toxicity labels
- Labels: Multiple toxicity categories (e.g., toxic, severe_toxic, obscene, threat, insult, identity_hate)

## Project Structure

```
Toxic-Comment-Class/
├── README.md                          # Project documentation
└── Toxic_Comment_Class.ipynb          # Main Jupyter notebook containing all code
```

## Dependencies

### Libraries Required

- **Data Processing**: pandas, numpy
- **Visualization**: seaborn, matplotlib
- **NLP**: nltk
- **Machine Learning**: scikit-learn
- **Utilities**: warnings

### Installation

Install required packages using:

```bash
pip install pandas numpy seaborn matplotlib nltk scikit-learn
```

Download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## Methodology

### 1. Data Loading & Exploration
- Load comment data from CSV
- Analyze dataset structure and missing values
- Visualize label distribution

### 2. Text Preprocessing
- **Cleaning**: Convert to lowercase, expand contractions, remove special characters
- **Stopword Removal**: Eliminate common English stopwords
- **Stemming**: Reduce words to their root forms using Snowball Stemmer

### 3. Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features with TF-IDF scoring

### 4. Model Building
- **Algorithm**: Logistic Regression with One-vs-Rest Classifier
- **Pipeline**: Integrated TF-IDF vectorizer with the classifier
- **Training**: Train on 80% of data, validate on 20% test set

### 5. Evaluation & Prediction
- Calculate accuracy on test set
- Make predictions on new, unseen comments
- Output probability scores for each toxicity category

## Usage

### Running the Notebook

Open the Jupyter notebook:

```bash
jupyter notebook Toxic_Comment_Class.ipynb
```

### Making Predictions

The notebook prompts for user input to classify comments:

```
Input: "Your Comment : [enter comment here]"
Output: 
toxic ------- [0/1]
severe_toxic ------- [0/1]
obscene ------- [0/1]
threat ------- [0/1]
insult ------- [0/1]
identity_hate ------- [0/1]
```

## Model Performance

The model achieves classification accuracy on the test dataset, identifying multiple toxicity categories per comment.

## Key Functions

- `remove_stopwords(text)`: Removes English stopwords from text
- `clean_text(text)`: Normalizes text by handling contractions and special characters
- `stemming(sentence)`: Applies stemming to reduce words to root forms
- `run_pipeline(pipeline, X_train, X_test, y_train, y_test)`: Trains and evaluates the model

## Results

The trained model can:
- Classify comments into multiple toxicity categories simultaneously
- Provide predictions for unseen comments
- Support binary classification (toxic/non-toxic) for each category

## Future Improvements

- Experiment with advanced models (Random Forest, Gradient Boosting, Neural Networks)
- Implement cross-validation for better model evaluation
- Add more sophisticated NLP techniques (word embeddings, LSTM)
- Create API endpoint for production deployment
- Add hyperparameter tuning
- Implement ensemble methods

## Author

Created as a machine learning classification project

## License

This project is open source and available for educational purposes.

---

**Note**: Ensure you have the training dataset (`train.csv`) in the `/content/` directory before running the notebook.
