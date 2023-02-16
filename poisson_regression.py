import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate random data
np.random.seed(0)
n = 1000
x = np.random.normal(0, 1, size=n)
y = np.random.poisson(np.exp(0.5 + 0.3 * x))

# Create pandas DataFrame
data = pd.DataFrame({'x': x, 'y': y})

# Fit Poisson regression model using statsmodels
model = sm.GLM(data['y'], sm.add_constant(data['x']), family=sm.families.Poisson())
result = model.fit()

# Print the model summary
print(result.summary())




####################################################################################
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate some random data
n_samples = 1000

# Independent variables
age = np.random.normal(loc=40, scale=10, size=n_samples)
income = np.random.normal(loc=50000, scale=10000, size=n_samples)
debt = np.random.normal(loc=20000, scale=5000, size=n_samples)

# Dependent variable
default_prob = np.exp(-0.5 + 0.05 * age + 0.1 * income + 0.02 * debt) / (1 + np.exp(-0.5 + 0.05 * age + 0.1 * income + 0.02 * debt))
defaults = np.random.binomial(1, default_prob)
data = pd.DataFrame({'Age': age, 'Income': income, 'Debt': debt, 'Defaults': defaults})

from sklearn.model_selection import train_test_split

X = data[['Age', 'Income', 'Debt']]
y = data['Defaults']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import statsmodels.api as sm

# Fit the Poisson regression model
poisson_model = sm.Poisson(y_train, sm.add_constant(X_train)).fit()

# Print the model summary
print(poisson_model.summary())
