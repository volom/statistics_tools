import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_breuschpagan


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

# calculate the autocorrelation function (ACF)
acf = sm.tsa.stattools.acf(model.resid)

# calculate the partial autocorrelation function (PACF)
pacf = sm.tsa.stattools.pacf(model.resid)

# perform the Ljung-Box test for autocorrelation
lb = acorr_ljungbox(model.resid, lags=[10], boxpierce=False)

# perform the Breusch-Godfrey test for autocorrelation
bg = het_breuschpagan(model.resid, model.model.exog)



# print the test statistic to the console
print("Durbin-Watson test statistic: ", dw)
print("Autocorrelation Function (ACF): ")
plot_acf(model.resid)
plt.show()
print("Partial Autocorrelation Function (PACF): ")
plot_pacf(model.resid)
plt.show()
print("Ljung-Box test results: ", lb)
print("Breusch-Godfrey test results: ", bg)

