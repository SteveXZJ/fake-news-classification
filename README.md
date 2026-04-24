# Fake News Detection with TF-IDF and Logistic Regression

This project builds a text classification model to distinguish between real and fake news articles using natural language processing and machine learning.

## Project Overview
The workflow includes:
- loading and preparing the dataset
- combining article title and body text
- training a TF-IDF + Logistic Regression baseline model
- evaluating performance with classification metrics and a confusion matrix
- interpreting important keywords
- testing the model on a real-world news article outside the original dataset

The goal is not only to build an accurate classifier, but also to better understand how the model makes decisions and how well it generalizes beyond the training data.

## Dataset
This project uses the Kaggle **Fake and Real News Dataset**(Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), which is distributed as two CSV files:
- `Fake.csv`
- `True.csv`

Each article includes fields such as:
- `title`
- `text`
- `subject`
- `date`

To frame the problem as binary classification:
- fake news articles were labeled as `1`
- real news articles were labeled as `0`

## Method
The baseline model uses:
- **TF-IDF vectorization**
- **Logistic Regression**

The article `title` and `text` fields are combined into a single `content` feature, which is then used as the model input.

## Results
Baseline performance on the test set:
- **Accuracy:** 0.9902
- **Precision:** 0.99
- **Recall:** 0.99
- **F1-score:** 0.99

The confusion matrix shows that the model correctly classifies the vast majority of both real and fake articles, with only a very small number of misclassifications.

## Feature Interpretation
The model associates fake news more strongly with sensational or attention-grabbing language, including terms such as `video`, `watch`, `breaking`, and `featured image`.

By contrast, real news is more strongly associated with formal reporting language such as `reuters`, `said`, `washington`, and weekday references like `tuesday` and `wednesday`.

This suggests that the model is learning not only topical patterns, but also differences in writing style and source conventions.

## Real-World Test
The model was also tested on an external Wall Street Journal article that was not part of the original Kaggle dataset.

It classified the article as **Real** with:
- **Real probability:** 0.8255
- **Fake probability:** 0.1745

This suggests that the model can generalize beyond the original dataset to some extent, although it may still rely partly on stylistic patterns found in professional news writing.

## Repository Contents
- `fake_news_classification.ipynb` — full notebook containing preprocessing, model training, evaluation, interpretation, and external testing
- `requirements.txt` — Python dependencies for the project

## Possible Next Steps
- test more real-world articles from different news sources
- compare results after removing source-identifying language
- try additional models such as Linear SVM or DistilBERT
