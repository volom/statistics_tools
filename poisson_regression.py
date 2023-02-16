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
