
# Loan Default Prediction (Credit Risk)

This project is a machine learning model that predicts whether a loan will default based on various features such as income, credit score, employment status, and other borrower characteristics. The goal is to help financial institutions identify high-risk borrowers and take preventive actions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Modeling](#modeling)
- [Challenges](#challenges)
- [Results](#results)
- [Requirements](#requirements)
- [Contributors](#contributors)

## Overview

Loan Default Prediction aims to use machine learning models (Keras and PyTorch) to classify whether a loan will default or not. The project includes:
- Data preprocessing and handling of imbalanced datasets.
- Exploratory Data Analysis (EDA) and feature engineering.
- Model training using both **Keras (TensorFlow)** and **PyTorch** frameworks.
- Evaluation and comparison of model performance using metrics like AUC-ROC, precision, recall, F1-score, and accuracy.

## Project Structure

```
loan-default-prediction/
│
├── data/                 # Folder for datasets (cleaned and backup data)
├── models/               # Folder for saved model files
├── notebooks/            # Jupyter notebooks for experiments
├── raw/              	  # Folder for raw data
│
├── README.md             # Project README
```

## Data

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default?resource=download). It contains 255,347 rows and 18 features, including various borrower attributes such as:
- **Income**: Annual income of the borrower.
- **LoanAmount**: Amount of the loan requested.
- **CreditScore**: Borrower's credit score.
- **EmploymentType**: Employment status of the borrower.
- **Education**: Highest education level attained.
- **MaritalStatus**: Marital status of the borrower.
- **HasMortgage**: Whether the borrower has an existing mortgage.
- **Default**: Target variable indicating if the borrower defaulted on the loan.

Data preprocessing steps included:
- Handling missing values.
- One-hot encoding of categorical features.
- Feature scaling for neural networks.
- Dealing with class imbalance using resampling techniques (SMOTE and undersampling).

## Modeling

Two machine learning models were implemented and compared:
- **Keras (TensorFlow)**: A neural network model with 2 hidden layers of 256 neurons each, using the ReLU activation function and dropout for regularization.
- **PyTorch**: A similar neural network architecture implemented in PyTorch for performance comparison.

### Key Model Parameters:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, AUC-ROC, Precision, Recall

### Handling Imbalanced Data:
The dataset was highly imbalanced, with far fewer loan defaults than non-defaults. To address this:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: To oversample the minority class (loan defaults).
- **Undersampling**: To reduce the number of samples from the majority class.

## Challenges

During the project, we initially encountered difficulties with model performance, where both Keras and PyTorch models were producing **AUC-ROC scores of 0.5**, indicating that the models were effectively guessing randomly. 

To address this, we focused on **feature engineering**, which involved:
- Transforming skewed features.
- Creating new interaction features, such as Debt-to-Income ratio.
- Resampling techniques to handle class imbalance more effectively.

These steps helped improve the models' performance, leading to significantly better results, particularly in predicting loan defaults.

## Results

### Keras Model:
- **AUC-ROC**: 0.852
- **Accuracy**: 76.6%
- **Default Prediction Accuracy**: 90.04%

### PyTorch Model:
- **AUC-ROC**: 0.850
- **Accuracy**: 76.4%
- **Default Prediction Accuracy**: 90.27%

Both models showed strong performance, with high recall for predicting defaults, ensuring that the model captures most of the actual defaults.

### Confusion Matrix (Keras):
```
[[28513 16626]  # Non-defaults
 [ 4497 40642]]  # Defaults
```

### Confusion Matrix (PyTorch):
```
[[28217 16922]  # Non-defaults
 [ 4394 40745]]  # Defaults
```

## Requirements

The project uses Python 3.9 and the following packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `torch`
- `matplotlib`
- `imbalanced-learn`


## Contributors

- **Dardeep Somel**
