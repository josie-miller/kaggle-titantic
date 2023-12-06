#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:12:29 2023
Sex, Pclass, Age, Fare, Embarked, is_alone, name_title
X = df[['Sex', 'Pclass', 'Age', 'Fare', "Embarked', 'is_alone', name_title']]
@author: josephinemiller

df['name_title'] = df['Name'].str.replace('.* ([A-Z][a-z]+)\..*', "\\1", regex=True)
print((df['name_title']))

df['name_title'] = df['name_title'].replace(['Countess','Capt', 'Col','Don', 'Major', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['name_title'] = df['name_title'].replace('Mlle', 'Miss')
df['name_title'] = df['name_title'].replace('Ms', 'Miss')
df['name_title'] = df['name_title'].replace('Mme', 'Mrs')

df['name_title'] = df['name_title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Lady": 5, "Dr": 6, "Rev": 7, "Rare": 8})
df['name_title'] = df['name_title'].fillna(0)


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

file_path = '/Users/josephinemiller/Downloads/train.csv'
df = pd.read_csv(file_path, index_col=False)

unique_values = df['Embarked'].unique()
print(unique_values)


df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})

# Assuming 'Ticket' column exists in the DataFrame df
df['Ticket'] = df['Ticket'].str.extract('(\d+)')

# Convert the 'Ticket' column to numeric
df['Ticket'] = pd.to_numeric(df['Ticket'], errors='coerce')

# Drop rows where 'Ticket' is NaN after conversion

df['family_size'] = (df['SibSp'] + df['Parch'] + 1)
print(df['family_size'])

# Whether or not a passenger is alone
df['is_alone'] = (df['family_size'] == 1).astype('int')
print(df['is_alone'])

#df['name_title'] = df['Name'].str.replace('.* ([A-Z][a-z]+)\..*', "\\1", regex=True)
#print((df['name_title']))

# Calculate the 50th percentile (median) of the 'Age' column
fill_age = np.percentile(df['Age'].dropna(), 50.0)

if np.isnan(fill_age):
    fill_age = df['Age'].mean()

# Replace NaN values in the 'Age' column with the calculated fill_age
df['Age'] = df['Age'].fillna(value=fill_age)

df = df.drop('Cabin', axis=1)
df = df.drop('Name', axis=1)
#df = df.drop('name_title', axis=1)

print(df.isnull().sum())
# Ticket has 4 N/A, Embarked has 2. They are dropped below.
df = df.dropna()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
#NOTE THIS
df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})

column_types = df.dtypes

# Display the data types of each column
print('IMPORTANT')

print(column_types)

# Split data into features (X) and target (Y)
X = df.iloc[:, [i for i in range(len(df.columns)) if i != 1]]
y = df[df.columns[1]]

# Check for NaN or infinite values in the features after preprocessing
if X.isnull().values.any() or not np.isfinite(X.select_dtypes(include=[np.number])).all().all():
    print("Error: X contains NaN or infinite values.")
    
    # Handle non-numeric columns
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns

        # Add handling for non-numeric columns if needed, such as encoding or dropping
        
    # Handle numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())  # Replace NaN with mean, adjust as needed
else:
    print("No NaN or infinite values found in X.")


print(X.head())
print(y.head())

# Descriptive statistics
print(X.describe())

# Histograms
num_columns = len(X.columns)
num_rows = int(np.ceil(num_columns / 2))  # Adjusted for the number of columns

plt.figure(figsize=(12, 10))
for i, column in enumerate(X.columns):
    plt.subplot(num_rows, 2, i+1)  # Adjusted the subplot dimensions
    plt.hist(X[column].dropna(), bins=20)  # Drop NaN values for histograms
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = df.corr(numeric_only=True)  # Specify numeric_only parameter
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter plots
for column in X.columns:
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(X[column]):
        plt.scatter(X[column], y)
        plt.xlabel(column)
        plt.ylabel('Y house price of unit area')
        plt.show()

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    included = list(initial_list)

    while True:
        changed = False

        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)

        for new_column in excluded:
            model_data = sm.add_constant(X[included + [new_column]]).copy()
            
            # Convert non-numeric columns to numeric
            non_numeric_columns = model_data.select_dtypes(exclude=[np.number]).columns
            model_data[non_numeric_columns] = model_data[non_numeric_columns].astype('category').apply(lambda x: x.cat.codes)
            
            model = sm.OLS(y, model_data).fit()
            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature:30} with p-value {best_pval:.6f}')

        # Backward step
        model_data = sm.add_constant(X[included]).copy()
        
        # Convert non-numeric columns to numeric
        non_numeric_columns = model_data.select_dtypes(exclude=[np.number]).columns
        model_data[non_numeric_columns] = model_data[non_numeric_columns].astype('category').apply(lambda x: x.cat.codes)
        
        model = sm.OLS(y, model_data).fit()
        
        # Use all coefficients except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature:30} with p-value {worst_pval:.6f}')

        if not changed:
            break

    return included


result = stepwise_selection(X, y)

print('resulting features:')
print(result)

plt.figure(figsize=(6, 9))


# Split the data into training and testing sets using all columns in X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the xgboost model normally using the scikit-learn API
xgb_model = XGBClassifier(max_depth=11,
                          learning_rate=0.1,
                          n_estimators=500,
                          subsample=0.75,
                          colsample_bylevel=1,
                          colsample_bytree=1,
                          scale_pos_weight=1.5,
                          reg_lambda=1.5,
                          reg_alpha=5,
                          n_jobs=8,
                          random_state=42,
                          use_label_encoder=False,
                          verbosity=0)

# Train the model
xgb_model.fit(X_train, y_train)

# Get the prediction of the model on the training data
y_train_pred = xgb_model.predict(X_train)

# Print some training predictions
df_train = pd.DataFrame({'Survived': y_train, 'prediction_xgb': y_train_pred})
print(df_train.head())

# Get feature importances
importances_sorted = xgb_model.feature_importances_
ind = np.argsort(importances_sorted)[::-1]
features_sorted = np.array(X.columns)[ind]  # Use all columns in X
# Plot feature importances
plt.barh(y=range(len(X.columns)), width=importances_sorted[ind], height=0.2)
plt.title('Gain')
plt.yticks(ticks=range(len(X.columns)), labels=features_sorted)
plt.gca().invert_yaxis()
plt.show()
