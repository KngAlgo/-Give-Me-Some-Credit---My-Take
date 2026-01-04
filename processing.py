import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("cs-training.csv")
test_data = pd.read_csv("cs-test.csv")

#---Feature Engineering

# create feature that says if borrower has high debt ratio

train_data['HighDebtRatio'] = (train_data['DebtRatio'] > 1.0).astype(int)
test_data['HighDebtRatio'] = (test_data['DebtRatio'] > 1.0).astype(int)

# clip debt ratios that are extremely high

train_data['DebtRatio'] = train_data['DebtRatio'].clip(upper=2.0)
test_data['DebtRatio'] = test_data['DebtRatio'].clip(upper=2.0)

# Create feature if borrower has missing income (NA value)

train_data['IncomeMissing'] = train_data['MonthlyIncome'].isna().astype(int)
test_data['IncomeMissing'] = test_data['MonthlyIncome'].isna().astype(int)

# Fill NA values for monthly income with median

train_median_income = train_data['MonthlyIncome'].median()
train_data['MonthlyIncome'].fillna(train_median_income, inplace=True)
test_data['MonthlyIncome'].fillna(train_median_income, inplace=True)

# Fille NA values for num dependents with median

train_data['NumberOfDependents'].fillna(train_data['NumberOfDependents'].median(), inplace=True)
test_data['NumberOfDependents'].fillna(train_data['NumberOfDependents'].median(), inplace=True)

#---Scaling

scaler = StandardScaler()

features_to_scale = ['age', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans']

train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])


def create_features_labels(train_data, test_data):
    x_train = train_data.drop(columns=["SeriousDlqin2yrs"])
    y_train = train_data['SeriousDlqin2yrs']

    x_test = test_data.drop(columns=["SeriousDlqin2yrs"])
    buffer = test_data['SeriousDlqin2yrs']

    return x_train, y_train, x_test, buffer

x_train, y_train, x_test, buffer = create_features_labels(train_data, test_data)