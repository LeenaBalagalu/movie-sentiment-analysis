# 🎬 Movie Review Sentiment Analysis using NLP & ML

📌 **Tools Used**: Python, Scikit-learn, NLTK, LIWC, VADER, XGBoost, RNN  
📁 **Dataset**: [Kaggle Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data) (156,000+ phrases)  
📈 **Goal**: Classify sentiment in movie review phrases using a range of NLP features and machine learning models

---

## 📊 Project Overview

This project explores the classification of movie review phrases into 5 sentiment categories ranging from "Very Negative" to "Very Positive". We experimented with a full NLP pipeline, including:

- Text preprocessing (tokenization, stopword removal, negation handling)
- Feature engineering with unigrams, bigrams, POS tagging, subjectivity scores, LIWC, and VADER lexicons
- Model training with various classifiers: Naïve Bayes, Logistic Regression, Random Forest, XGBoost, and Recurrent Neural Networks (RNN)
- Model performance comparison using F1-score, Precision, Recall, and Accuracy

---

## 🧠 Key Features Engineered

- **Unigrams & Bigrams**: Tokenized features for phrase representation  
- **Negation Handling**: Captured sentiment-flipping phrases like "not good"  
- **POS Tagging**: Used to extract adjectives/adverbs linked to emotional intensity  
- **LIWC Lexicons**: Used to analyze cognitive/emotional word groups  
- **VADER Lexicons**: Measured sentiment intensity in short phrases  
- **Subjectivity Analysis**: Separated objective from subjective phrases

---

## 🤖 Models Evaluated

| Model              | Accuracy | F1 Score | Notes |
|--------------------|----------|----------|-------|
| Naïve Bayes        | 54.0%    | 0.540    | Performs well with unigrams but struggles with complex features |
| Logistic Regression| 59.0%    | 0.603    | Best traditional model, handles rich feature sets better |
| XGBoost            | 58.4%    | 0.541    | Strong with structured data, but affected by class imbalance |
| Random Forest      | 45.1%    | 0.425    | Poor generalization with minority classes |
| **RNN**            | **72.6%**| **0.700**| Best overall; captures sequence and context well |

---

## 📊 Results Summary

- RNN achieved the highest F1 score (0.825) for neutral class and best overall performance
- Class imbalance significantly reduced recall for negative and positive sentiment classes
- Logistic Regression showed reliable results among traditional models
- Combining features improved results by over 12% compared to using unigrams alone

---

## 📌 Key Takeaways

- Rich feature engineering improves NLP sentiment models — especially with subjectivity, negation, and POS tagging
- Simple models like Naive Bayes struggle with complex feature sets
- RNNs are powerful but require tuning and are computationally expensive
- Addressing class imbalance is essential for better performance on minority sentiment classes

