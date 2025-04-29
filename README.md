### cardiovascular-disease-predictions-leveraging-supervised-and-ensemble-machine-learning-models

## Project Overview 

This research project focuses on building and evaluating multiple machine learning models to predict cardiovascular disease based on patient health attributes. By combining two diverse datasets, the model benefits from a richer, more generalized training environment, ultimately improving predictive accuracy.

## 📁 Datasets

1. Primary Dataset (Mendeley Data)
   
Source: A multispecialty hospital in India via Mendeley Data ( https://data.mendeley.com/datasets/dzz48mvjht/1/files/e4a4a2de-2783-4ea8-9958-0fc3c82cadd4)

Records: 1,000 individuals

Features: 12 features including:

Age

Blood Pressure

Cholesterol Levels

Chest Pain Type

Gender

Other lifestyle and health-related factors

2. Secondary Dataset (UCI Repository)
 
Source: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/45/heart+disease)

Records: 303 individuals

Features: Similar attributes to the primary dataset

These datasets were merged to enhance the diversity and size of the training data, which supports better generalization and robust model performance.

##  Models Implemented

# Supervised Machine Learning Models:

Logistic Regression

K-Nearest Neighbors (KNN)

Naive Bayes

Support Vector Machine (SVM)

# Ensemble Learning Models:

Gradient Boosting Classifier

Bagging Classifier (with Decision Tree)

Stacking Classifier (Random Forest + Logistic Regression)

## Evaluation Metrics

The models were evaluated using the following metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

ROC Curves

Visualizations were provided to compare model performances across all metrics.

##Key Findings

- Ensemble models, particularly Stacking Classifier, consistently outperformed individual models.

- Gradient Boosting and Bagging also yielded high F1-scores and ROC-AUC values.

- SVM showed variable performance across datasets and required class balancing using SMOTE for better results.

- Models trained on the combined dataset demonstrated improved generalizability and accuracy.

## Technologies Used

Python (Pandas, NumPy)

Scikit-learn

Seaborn & Matplotlib

Mlxtend (for stacking)

SMOTE (for handling class imbalance)






