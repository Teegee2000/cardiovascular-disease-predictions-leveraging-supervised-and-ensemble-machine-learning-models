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

# Supervised Machine Learning Models for mendeley dataset:

Logistic Regression (68%)

K-Nearest Neighbors (KNN) (52%)

Naive Bayes  (81%) 

Support Vector Machine (SVM) (51%)

# Ensemble Learning Models for mendeley dataset:

Gradient Boosting Classifier (97%)

Bagging Classifier (with Decision Tree) (96%)

Stacking Classifier (Random Forest + Logistic Regression) (98%)

# Supervised Machine Learning Models for mendeley dataset:

Logistic Regression (70%)

K-Nearest Neighbors (KNN) (69%)

Naive Bayes  (87%) 

Support Vector Machine (SVM) (89%)

# Ensemble Learning Models for mendeley dataset:

Gradient Boosting Classifier (77%)

Bagging Classifier (with Decision Tree) (88%)

Stacking Classifier (Random Forest + Logistic Regression) (85%)


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

##  Observations & Insights
-When working with larger and more diverse datasets, ensemble models such as Gradient Boosting, Bagging, and Stacking consistently outperformed individual supervised models.

-This is primarily because ensemble models have a higher capacity to capture complex patterns and generalize better by combining the strengths of multiple learners.

-Individual supervised models like Logistic Regression, SVM, KNN, and Naive Bayes performed reasonably well, but they often struggled with overfitting or underfitting on the larger dataset.

-In contrast, ensemble models demonstrated robustness, higher precision, and better recall, making them more suitable for real-world healthcare applications where data can be high-dimensional and imbalanced.




