# Insurance Cross-Sell Prediction

## Project Overview

This project aims to predict whether an existing insurance customer will subscribe to a vehicle insurance policy.

The dataset is highly imbalanced (~12% positive class), making recall optimization crucial to avoid missing potential customers.

The project includes:
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Feature engineering
- Handling imbalanced data
- Logistic Regression modeling
- Threshold tuning
- Model comparison


## Business Objective

The goal is to identify customers likely to accept a vehicle insurance offer.

From a business perspective:
- False Negatives (missed potential customers) are costly
- False Positives are acceptable to some extent

Therefore, the main objective is maximizing recall while keeping acceptable precision.


## Dataset

The dataset contains 381,109 observations with demographic and insurance-related features.

Target variable:
- 1 → Customer subscribed
- 0 → Customer refused

The dataset is strongly imbalanced (≈12% positive class).


## Exploratory Data Analysis

Key findings:
- Age and Vehicle_Age show strong correlation.
- Vehicle_Damage and Previously_Insured are strongly correlated with Response.
- Annual_Premium is statistically significant.
- No missing values detected.
- Vintage is not significantly correlated with the target.

Statistical tests used:
- Chi-squared test
- Cramer's V
- T-test
- ANOVA


## Preprocessing

- Dropped ID column
- Encoded categorical variables
- Grouped rare categories in Region_Code and Policy_Sales_Channel
- One-Hot Encoding
- Standardized numerical features
- Stratified train-test split
- Resampling techniques (over + under sampling)


## Models Implemented

### Baseline Logistic Regression
Very low recall, biased toward majority class.

### Logistic Regression (class_weight="balanced")
Recall ≈ 94%  
Precision ≈ 28%  
Best business-aligned model.

### Logistic Regression with Resampling
Similar recall, slight train-test gap.

### Logistic Regression with L1 Regularization
No significant improvement.


## Final Model Performance (Test Set)

| Metric | Value |
|--------|--------|
| Recall | ~0.94 |
| Precision | ~0.28 |
| F1 Score | ~0.43 |
| AUC | (insert your value) |


## Key Learning Points

- Handling imbalanced datasets
- Business-driven metric selection
- Threshold optimization
- Regularization techniques
- Model evaluation beyond accuracy


## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels, Imbalanced-learn


