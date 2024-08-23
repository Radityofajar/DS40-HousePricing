# DS40-HousePricing Prediction Project

This project aims to predict house prices using various features of a dataset obtained from Kaggle. The project involves data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation using Python libraries.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Data Preprocessing and Feature Engineering](#feature-engineering)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Results](#results)

## Project Overview

The goal of this project is to build a machine learning model that accurately predicts house prices based on a variety of features such as the lot area, year built, overall quality, and more. This project is implemented in Python using Jupyter Notebook, leveraging libraries like pandas, NumPy, matplotlib, seaborn, and scikit-learn.

## Dataset

The dataset used in this project is from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) competition on Kaggle.

- **Train Dataset:** `ds_project3_train.csv`
- **Test Dataset:** `ds_project3_test.csv`

## Installation

To run this project, you will need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
or just simply 
```bash
pip install -r requirements.py
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
``` 
2. Navigate to the repository project directory:
```bash
cd your-repo-name
``` 
3. Open the Jupyter Notebook:
```bash
jupyter notebook DS40_Project3.ipynb
```

## Data Processing
The preprocessing steps include:

- Importing necessary libraries.
- Loading the train and test datasets.
- Selecting relevant features for the analysis.
- Handling missing values and encoding categorical variables.

## Exploratory Data Analysis
The notebook provides an in-depth analysis of the dataset using visualizations and statistical summaries. Key steps include:

- Understanding the distribution of the target variable (SalePrice).
- Visualizing the relationships between different features.

## Feature Engineering
Feature engineering involves transforming and encoding data to improve model performance. Steps include:

- Handling missing values.
- Encoding categorical variables.
- Feature selection.

## Model Training and Evaluation
The project uses regression models to predict house prices. Each model performance is evaluated using metrics like MAE, MSE, RMSE, and R-squared. The model evaluate are Linear Regression, Lasso Regression, SVR, XGBoost, and ANN. 

## Results
The models developed in this project achieve a reasonable prediction accuracy on the test dataset. Further improvements can be made by tuning hyperparameters and exploring additional features.
