# Heart Disease Prediction using Machine Learning

This project aims to predict heart disease using various machine learning algorithms. By analyzing patient data, the project provides accurate predictions to help in early detection and management of heart disease.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Steps](#detailed-steps)
- [Algorithms Used](#algorithms-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Heart disease is a leading cause of death worldwide. Early detection and management can significantly improve patient outcomes. This project utilizes machine learning techniques to predict the presence of heart disease in patients based on various health metrics.

## Dataset

The dataset used in this project is the 'cardio_train.csv' file, which contains 70,000 rows and 13 columns. The columns include:

- id
- age
- gender
- height
- weight
- ap_hi (systolic blood pressure)
- ap_lo (diastolic blood pressure)
- cholesterol
- gluc (glucose)
- smoke
- alco (alcohol intake)
- active (physical activity)
- cardio (presence of heart disease)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install the required libraries using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Open the Jupyter notebook or Google Colab:

```bash
jupyter notebook HeartDisease.ipynb
```

3. Run the notebook cells to execute the code and see the results.

## Detailed Steps

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
```

- Import necessary libraries for data manipulation, visualization, and machine learning.

### Step 2: Load Dataset

```python
data = pd.read_csv('cardio_train.csv', sep=';')
data.head()
```

- Load the dataset and display the first few rows.

### Step 3: Data Preprocessing

#### Handle Missing Values

```python
data.isnull().sum()
```

- Check for missing values in the dataset.

#### Data Cleaning

```python
data['age'] = data['age'] / 365.25  # Convert age from days to years
data = data.drop(['id'], axis=1)  # Drop the 'id' column
```

- Convert age from days to years and drop the 'id' column.

#### Feature Engineering

```python
data['BMI'] = data['weight'] / (data['height']/100)**2  # Calculate BMI
data = pd.get_dummies(data, columns=['cholesterol', 'gluc'], drop_first=True)  # One-hot encode categorical variables
```

- Create a new feature 'BMI' and one-hot encode categorical variables.

### Step 4: Train-Test Split

```python
X = data.drop('cardio', axis=1)
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Split the data into training and testing sets.

### Step 5: Feature Scaling

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- Scale the features for better performance of machine learning algorithms.

### Step 6: Model Training and Evaluation

#### Random Forest Classifier

```python
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, rf_preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, rf_preds))
```

- Train a Random Forest classifier and evaluate its performance.

#### XGBoost Classifier

```python
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)
print('XGBoost Accuracy:', accuracy_score(y_test, xgb_preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, xgb_preds))
```

- Train an XGBoost classifier and evaluate its performance.

## Algorithms Used

The following machine learning algorithms were used in this project:

- Random Forest
- XGBoost

## Results

The performance of each algorithm was evaluated using the accuracy score. The results are as follows:

- Random Forest: [Accuracy]
- XGBoost: [Accuracy]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

