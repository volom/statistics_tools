import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Fit a linear regression model on the reduced data
model = LinearRegression()
model.fit(X_train_reduced, y_train)

# Predict the target values for the test data
y_pred = model.predict(X_test_reduced)

# Evaluate the model's performance
mse = np.mean((y_test - y_pred)**2)
print('Mean Squared Error:', mse)
