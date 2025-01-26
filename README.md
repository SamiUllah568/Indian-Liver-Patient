# Indian Liver Patient Prediction


**Author:** SamiUllah568

## Table of Contents
1. [Context](#context)
2. [About the Dataset](#about-the-dataset)
3. [Import Necessary Libraries](#import-necessary-libraries)
4. [Load Dataset](#load-dataset)
5. [Data Preprocessing](#data-preprocessing)
    - [Rename Target Column](#rename-target-column)
    - [Class Value Replacement](#class-value-replacement)
    - [Dataset Information and Statistical Summary](#dataset-information-and-statistical-summary)
    - [Handling Missing Values](#handling-missing-values)
    - [Managing Duplicate Records](#managing-duplicate-records)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Class Distribution](#class-distribution)
    - [Gender Distribution Analysis](#gender-distribution-analysis)
    - [Gender Distribution by Class](#gender-distribution-by-class)
    - [Distribution of Numerical Features](#distribution-of-numerical-features)
    - [Distribution of Numerical Features by Class](#distribution-of-numerical-features-by-class)
    - [Boxplots for Alamine and Aspartate Aminotransferase](#boxplots-for-alamine-and-aspartate-aminotransferase)
7. [Outlier Removal Using the Interquartile Range (IQR) Method](#outlier-removal-using-the-interquartile-range-iqr-method)
    - [Boxplots After Removing Outliers](#boxplots-after-removing-outliers)
8. [Correlation Matrix of Numerical Features](#correlation-matrix-of-numerical-features)
9. [Feature Engineering](#feature-engineering)
    - [Splitting Features and Target Variable](#splitting-features-and-target-variable)
    - [One-Hot Encoding for 'Gender' Feature](#one-hot-encoding-for-gender-feature)
    - [Feature Scaling Using StandardScaler](#feature-scaling-using-standardscaler)
10. [Model Training](#model-training)
    - [Logistic Regression](#logistic-regression)
    - [XGBClassifier](#xgbclassifier)
    - [Random Forest](#random-forest)
11. [Conclusion](#conclusion)
12. [Further Applications](#further-applications)

## Context

Patients with liver disease have been continuously increasing due to excessive consumption of alcohol, inhalation of harmful gases, intake of contaminated food, pickles, and drugs. This dataset was used to evaluate prediction algorithms in an effort to reduce the burden on doctors.

## About the Dataset

This dataset contains 416 liver patient records and 167 non-liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This dataset contains 441 male patient records and 142 female patient records.

DataSet Link -- >>  https://www.kaggle.com/datasets/uciml/indian-liver-patient-records 

### Columns:

- Age of the patient
- Gender of the patient
- Total Bilirubin
- Direct Bilirubin
- Alkaline Phosphotase
- Alamine Aminotransferase
- Aspartate Aminotransferase
- Total Proteins
- Albumin
- Albumin and Globulin Ratio
- Class: field used to split the data into two sets (patient with liver disease -->> 1, or no disease --> 0)

## Import Necessary Libraries

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
sns.set_palette("muted")
```

## Load Dataset

```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("indian_liver_patient.csv")
df.head()
```

## Data Preprocessing

### Rename Target Column

```python
df.rename(columns={"Dataset": "Class"}, inplace=True)
```

### Class Value Replacement

```python
df["Class"].replace(2, 0, inplace=True)
```

### Dataset Information and Statistical Summary

```python
print(df.info())
print(df.describe())
```

### Handling Missing Values

```python
df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(df['Albumin_and_Globulin_Ratio'].mean())
```

### Managing Duplicate Records

```python
df.drop_duplicates(inplace=True)
```

## Exploratory Data Analysis

### Class Distribution

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=df["Class"], ax=ax[0], saturation=0.75, palette=['skyblue', 'lightcoral'])
ax[0].set_title("Patients with liver disease --> 1, No disease --> 0")
ax[1].pie(df["Class"].value_counts(), labels=df["Class"].value_counts().index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightpink'])
ax[1].set_title("Class Distribution (Pie Chart)")
plt.tight_layout()
plt.show()
```

### Gender Distribution Analysis

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=df["Gender"], ax=ax[0], saturation=0.75, palette=['skyblue', 'lightcoral'])
ax[0].set_title("Gender Distribution")
ax[1].pie(df["Gender"].value_counts(), labels=df["Gender"].value_counts().index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightpink'])
ax[1].set_title("Gender Distribution (Pie Chart)")
plt.tight_layout()
plt.show()
```

### Gender Distribution by Class

```python
sns.countplot(x=df["Gender"], hue=df["Class"], saturation=0.75, palette=['skyblue', 'lightcoral'])
plt.title("Gender Hue with Class", fontweight='bold')
plt.show()
```

### Distribution of Numerical Features

```python
fig, ax = plt.subplots(5, 2, figsize=(14, 25))
ax = ax.flatten()
for i, feature in enumerate(num_features.columns):
    sns.histplot(data=df, x=feature, kde=True, hue="Class", ax=ax[i], palette="Set2", alpha=0.6)
    ax[i].set_title(f'Distribution of {feature} by Class', fontsize=14)
plt.tight_layout()
plt.show()
```

### Boxplots for Alamine and Aspartate Aminotransferase

```python
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df["Class"], y=df["Alamine_Aminotransferase"])
plt.title('Alamine Aminotransferase by Class')
plt.subplot(1, 2, 2)
sns.boxplot(x=df["Class"], y=df["Aspartate_Aminotransferase"])
plt.title('Aspartate Aminotransferase by Class')
plt.tight_layout()
plt.show()
```

## Outlier Removal Using the Interquartile Range (IQR) Method

```python
def remove_outliers(data, column, group_by):
    cleaned_data = data.copy()
    for group, subset in cleaned_data.groupby(group_by):
        q1 = np.percentile(subset[column].dropna(), 25)
        q3 = np.percentile(subset[column].dropna(), 75)
        IQR = q3 - q1
        lower = q1 - 1.5 * IQR
        upper = q3 + 1.5 * IQR
        is_in_bounds = (cleaned_data[column] >= lower) & (cleaned_data[column] <= upper) & (cleaned_data[group_by] == group)
        cleaned_data = cleaned_data[is_in_bounds | (cleaned_data[group_by] != group)]
    return cleaned_data

df2 = remove_outliers(df, column="Albumin_and_Globulin_Ratio", group_by="Class")
df2 = remove_outliers(df2, column="Aspartate_Aminotransferase", group_by="Class")
```

### Boxplots After Removing Outliers

```python
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df2["Class"], y=df2["Alamine_Aminotransferase"])
plt.title('After Removing Outliers Alamine Aminotransferase')
plt.subplot(1, 2, 2)
sns.boxplot(x=df2["Class"], y=df2["Aspartate_Aminotransferase"])
plt.title('After Removing Outliers Aspartate Aminotransferase')
plt.tight_layout()
plt.show()
```

## Correlation Matrix of Numerical Features

```python
matrix = df2.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(matrix, cbar=True, square=True, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Scale'})
plt.title('Correlation between all the features', fontsize=16)
plt.tight_layout()
plt.show()
```

## Feature Engineering

### Splitting Features and Target Variable

```python
X = df2.drop("Class", axis=1)
y = df2["Class"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
```

### One-Hot Encoding for 'Gender' Feature

```python
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int8)
x_train_encoded = ohe.fit_transform(x_train[["Gender"]])
x_test_encoded = ohe.transform(x_test[["Gender"]])
x_train[["Gender"]] = x_train_encoded
x_test[["Gender"]] = x_test_encoded
```

### Feature Scaling Using StandardScaler

```python
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
x_train = pd.DataFrame(data=x_train, columns=X.columns)
x_test = pd.DataFrame(data=x_test, columns=X.columns)
```

## Model Training

### Logistic Regression

```python
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
train_log_predicted = logreg.predict(x_train)
test_log_predicted = logreg.predict(x_test)
logreg_score = round(logreg.score(x_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(x_test, y_test) * 100, 2)
print('Logistic Regression Training Score:', logreg_score)
print('Logistic Regression Test Score:', logreg_score_test)
print('Accuracy on train:', accuracy_score(y_train, train_log_predicted))
print('Confusion Matrix on train:', confusion_matrix(y_train, train_log_predicted))
print('Classification Report on train:', classification_report(y_train, train_log_predicted))
print('Accuracy on test:', accuracy_score(y_test, test_log_predicted))
print('Confusion Matrix on test:', confusion_matrix(y_test, test_log_predicted))
print('Classification Report on test:', classification_report(y_test, test_log_predicted))
sns.heatmap(confusion_matrix(y_test, test_log_predicted), annot=True, fmt="d")
plt.show()
```

### XGBClassifier

```python
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred_train = xgb.predict(x_train)
y_pred_test = xgb.predict(x_test)
print("===== Train Data Evaluation =====")
print(f"Accuracy Score: {roc_auc_score(y_train, y_pred_train):.2f}")
print("\nClassification Report:")
print(classification_report(y_train, y_pred_train))
print("\n===== Test Data Evaluation =====")
print(f"Accuracy Score: {roc_auc_score(y_test, y_pred_test):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title("Confusion Matrix (Test Data)", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.show()
```

### Random Forest

```python
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
rf_predicted = random_forest.predict(x_test)
random_forest_score = round(random_forest.score(x_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(x_test, y_test) * 100, 2)
print('Random Forest Score:', random_forest_score)
print('Random Forest Test Score:', random_forest_score_test)
print('Accuracy:', accuracy_score(y_test, rf_predicted))
print(confusion_matrix(y_test, rf_predicted))
print(classification_report(y_test, rf_predicted))
```

## Conclusion

Logistic Regression performed well on this dataset. However, it is not expected to get the same level of performance on bigger and denser datasets due to the small size and imbalance of this dataset.

## Further Applications

The dataset we worked on was very small, consisting of only 583 observations where some outliers were removed. The dataset was highly unbalanced, with positive records being three times the number of negative ones. Hence, even though we have obtained good scores on this dataset, the performance of the same models on similar but bigger datasets is expected to vary.
