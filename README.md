# 🧠 Sentiment Analysis on E-Commerce Product Reviews using Transformer Models

This project implements a sentiment classifier for e-commerce product reviews using modern NLP models, including **BERT**, **DistilBERT**, **ALBERT**, and **ELECTRA**, powered by the Hugging Face Transformers library.

We compare multiple models, apply preprocessing, conduct hyperparameter tuning using **Optuna**, and evaluate the models based on accuracy, precision, recall, and F1 score. The best performing model (BERT) is further optimized for final predictions.

## 🔍 Overview

This project addresses the binary sentiment classification problem (positive/negative) using real customer reviews. The full pipeline includes:
- Data cleaning & preprocessing
- Tokenization using Hugging Face AutoTokenizer
- Training multiple transformer models using `Trainer` API
- Evaluation based on validation metrics
- Hyperparameter tuning using Optuna
- Error analysis on misclassified samples

---

## 🚀 Models Compared

| Model      | Accuracy | F1 Score | Precision | Recall | Error Rate |
|------------|----------|----------|-----------|--------|------------|
| BERT       | 96.22%   | 96.23%   | 96.23%    | 96.22% | 3.78%      |
| DistilBERT | 94.92%   | 94.80%   | 94.77%    | 94.92% | 5.08%      |
| ELECTRA    | 96.00%   | 95.91%   | 95.91%    | 96.00% | 4.00%      |
| ALBERT     | 93.09%   | 92.86%   | 92.79%    | 93.09% | 6.91%      |

---

## 🧪 Tech Stack
- Python 3.x
- Hugging Face Transformers
- PyTorch
- Optuna (for hyperparameter tuning)
- Scikit-learn, Pandas
- Google Colab (for training environment)
