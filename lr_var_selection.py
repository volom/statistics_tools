# LASSO Selection
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Create the Lasso Regression model
model = Lasso(alpha=0.1)

# Fit the model to the data
model.fit(X, y)

# Predict target values
y_pred = model.predict(X)

# Print the coefficients of the model
print(model.coef_)



# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Create the Ridge Regression model
model = Ridge(alpha=0.1)

# Fit the model to the data
model.fit(X, y)

# Predict target values
y_pred = model.predict(X)

# Print the coefficients of the model
print(model.coef_)



# Recursive Feature Elimination (RFE)

from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Create the Linear Regression model
lr = LinearRegression()

# Use RFE to select the most important features
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X, y)

# Print the selected features
print(rfe.support_)


# Random Forest
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Create the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100)

# Train the model on the data
rf.fit(X, y)

# Use SelectFromModel to select the most important features
sfm = SelectFromModel(rf, threshold=0.25)
sfm.fit(X, y)
X_important = sfm.transform(X)

# Get the feature importances from the Random Forest model
importances = rf.feature_importances_

# Sort the importances in descending order
indices = np.argsort(importances)[::-1]

# Print the feature names and importances
print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature {}{:<20} ({:.4f})".format(f + 1, boston.feature_names[idx], "", importances[idx]))

# Create the Linear Regression model
lr = LinearRegression()

# Train the model on the important features
lr.fit(X_important, y)





# Stepwise Regression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the list of included features
included = []

# Initialize the model
model = LinearRegression()

# Set a threshold for the p-value
threshold = 0.05

# Loop through each feature
for i in range(X.shape[1]):
    
    # Get the current set of included features
    current = list(included)
    
    # Add the current feature to the set of included features
    current.append(X.columns[i])
    
    # Fit the model with the current set of features
    model.fit(X_train[current], y_train)
    
    # Get the p-value of the current feature
    p_value = model.pvalues_[-1]
    
    # If the p-value is greater than the threshold, remove the current feature
    if p_value > threshold:
        included.pop()
    
    # If the p-value is less than the threshold, add the current feature
    else:
        included.append(X.columns[i])

# Fit the final model with the selected features
model.fit(X_train[included], y_train)

