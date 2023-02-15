import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import statsmodels.api as sm

# load the Boston Housing dataset
boston = load_boston()

# create a pandas DataFrame with the feature and target data
df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# fit a linear regression model
X = df.drop('MEDV', axis=1)
y = df['MEDV']
model = sm.OLS(y, sm.add_constant(X)).fit()

# calculate the Durbin-Watson test statistic for autocorrelation
dw = sm.stats.stattools.durbin_watson(model.resid)

# print the test statistic to the console
print("Durbin-Watson test statistic:", dw)
