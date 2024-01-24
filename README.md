**Consolidated Project Summary: Customer Churn Prediction**

**Objective:**
The primary objective of this project is to predict customer churn, aiding businesses in proactively retaining customers. The project uses a structured dataset containing various customer features, and the target variable is binary, indicating whether a customer has churned (1) or not (0).

**Dataset:**
The dataset used for this project contains information about customers, including various features that may influence their likelihood of churning. The features could include demographic information, usage patterns, plan details, payment methods, and more. The dataset is assumed to be structured, and the target variable is typically binary, indicating whether a customer has churned (1) or not (0).

**Models Used:**
1. **Random Forest Classifier:**
   - A versatile ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and robustness.

2. **Logistic Regression:**
   - A simple yet effective linear model used for binary classification tasks. It models the probability of an instance belonging to a particular class.

3. **Support Vector Machine (SVM):**
   - A powerful algorithm for both classification and regression tasks. SVM aims to find the hyperplane that best separates the classes in the feature space.

4. **Gradient Boosting Classifier:**
   - An ensemble learning method that builds a sequence of weak learners (usually decision trees) and combines their predictions to create a strong learner.

5. **k-Nearest Neighbors (KNN):**
   - A non-parametric classification algorithm that assigns an instance to the class most common among its k nearest neighbors in the feature space.

**Models Results:**
1. **Random Forest Classifier:**
   - Best Hyperparameters: {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
   - Cross-Validation Accuracy: 79.4%
   - Test Set Accuracy: 78.95%
   - Precision (class 0): 79%, Recall (class 1): 0%

2. **Logistic Regression:**
   - Best Hyperparameters: {'C': 0.001}
   - Cross-Validation Accuracy: 79.41%
   - Test Set Accuracy: 78.95%
   - Precision (class 0): 79%, Recall (class 1): 0%

3. **SVM:**
   - Best Hyperparameters: {'C': 0.1, 'kernel': 'linear'}
   - Cross-Validation Accuracy: 79.41%
   - Test Set Accuracy: 78.95%
   - Precision (class 0): 79%, Recall (class 1): 0%

4. **Gradient Boosting:**
   - Best Hyperparameters: {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100}
   - Cross-Validation Accuracy: 79.44%
   - Test Set Accuracy: 78.9%
   - Precision (class 0): 79%, Recall (class 1): 0.24%

5. **KNN:**
   - Best Hyperparameters: {'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}
   - Cross-Validation Accuracy: 77.06%
   - Test Set Accuracy: 77.0%
   - Precision (class 0): 79%, Recall (class 1): 0.06%

**General Observations:**
- Models exhibit challenges in predicting class 1 (churn), reflected in low recall values for this class across all models.
- Precision for class 0 (non-churn) is generally high, indicating good performance in identifying customers who do not churn.
- A class imbalance issue is evident, impacting model performance as they tend to predict the majority class (non-churn).

**Recommendations:**
- Address class imbalance through oversampling or alternative metrics like F1-score.
- Consider feature engineering or acquiring additional features to better capture churn-related patterns.
- Experiment with advanced models or ensemble techniques for improved predictive performance.

**Consolidated Insights:**
- All models struggle to identify customers at risk of churning (low recall for class 1).
- Precision for non-churners is consistently high across models.
- Addressing the class imbalance and further refining the models are critical for enhancing performance.
- Future iterations may involve advanced modeling techniques and feature engineering to improve the models' ability to identify customers likely to churn.
