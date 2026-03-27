# CreditCardFraudDetection: Predictive Modeling for Financial Security

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Library-ImbLearn-blueviolet.svg)](https://imbalanced-learn.org/)


## Executive Summary
**CreditCardFraudDetection** is a comprehensive machine learning pipeline designed to identify fraudulent transactions with high precision. In the financial sector, fraud detection is characterized by extreme **class imbalance**—where legitimate transactions vastly outnumber fraudulent ones. 

This project implements advanced resampling techniques and cost-sensitive learning to navigate the "Accuracy Paradox," ensuring the model prioritizes **Recall** (catching fraud) without overwhelming the system with False Positives.

---

## Technical Methodology

### 1. Data Characterization & PCA
The dataset utilized consists of transactions made by European cardholders. Due to confidentiality, features $V_1, V_2, \dots, V_{28}$ are the result of a **Principal Component Analysis (PCA)** transformation. 
* **Challenge**: Non-interpretable features require a heavy reliance on statistical distributions and correlation matrices rather than domain-specific intuition.

### 2. Addressing Class Imbalance
To prevent the model from simply predicting the majority class, this project explores:
* **SMOTE (Synthetic Minority Over-sampling Technique)**: Generating synthetic examples of the minority class to balance the training set.
* **Under-sampling**: Reducing the majority class to prevent gradient bias during training.
* **Stratified K-Fold Cross-Validation**: Ensuring each fold maintains the original class distribution to provide a robust estimate of model performance.

### 3. Algorithm Benchmarking
We evaluate multiple classifiers to determine the optimal decision boundary:
* **Logistic Regression**: A baseline for linear separability.
* **Random Forest / XGBoost**: Ensemble methods to capture non-linear interactions between PCA components.
* **Isolation Forests**: An anomaly detection approach to identify outliers in high-dimensional space.

---

## Evaluation Metrics (Beyond Accuracy)

In fraud detection, **Accuracy is a deceptive metric**. A model predicting "Not Fraud" 100% of the time would achieve $>99\%$ accuracy but fail its objective. We instead focus on:

* **AUPRC (Area Under the Precision-Recall Curve)**: The gold standard for imbalanced datasets, emphasizing the trade-off between Precision and Recall.
* **F1-Score**: The harmonic mean of precision and recall.
* **Confusion Matrix Analysis**: Specifically minimizing **False Negatives** (undetected fraud) while maintaining a tolerable **False Positive Rate** to protect user experience.

---

## Getting Started

### Prerequisites
* Python 3.9+
* Pandas, NumPy, Scikit-Learn
* Imbalanced-learn (`imblearn`)

### Installation
git clone [https://github.com/AKDev32/CreditCardFraudDetection.git](https://github.com/AKDev32/CreditCardFraudDetection.git)
cd CreditCardFraudDetection
pip install -r requirements.txt


---

## Educational Objectives

Through the development of this predictive pipeline, I have conducted empirical investigations into:

* **The Accuracy Paradox**: Mathematically demonstrating why standard accuracy metrics are insufficient for skewed distributions (0.17% minority class) and justifying the shift toward **AUPRC** and **F1-Score**.
* **Feature Engineering & Normalization**: Analyzing the necessity of robust scaling for the "Time" and "Amount" features to ensure they do not disproportionately bias the objective function relative to the PCA-transformed variables.
* **Cost-Sensitive Learning**: Evaluating the economic trade-off between **False Positives** (customer friction/operational cost) and **False Negatives** (direct financial loss from undetected fraud).

---

## Future Work

To enhance the model's robustness and utility, the following research directions are proposed:

1.  **Unsupervised Anomaly Detection**: Implementing **Deep Autoencoders** to learn a compressed representation of "normal" transactions, allowing the model to detect "zero-day" fraud patterns that deviate from the latent distribution.
2.  **Real-Time Inference Pipeline**: Developing a high-concurrency REST API using **FastAPI** to demonstrate how the model would score incoming transaction streams in a production environment.
3.  **SHAP/LIME Integration**: Applying Model-Agnostic Explanations to provide interpretability for the model's decisions, which is a critical requirement for regulatory compliance in fintech.

---

**Author:** [Aman Kumar / AKDev32]  
**Academic Focus:** Data Science / Financial Engineering
