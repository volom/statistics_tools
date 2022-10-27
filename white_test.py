# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 18:40:34 2022


#from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
import pandas as pd
import numpy as np

significance_level = 0.05
df_name = "4white_model2021.xlsx"

df = pd.read_excel(df_name)


y = df.iloc[:, [0]]
x = df.iloc[:, 1:]

x = sm.add_constant(x)

#fit regression model
model = sm.OLS(y, x).fit()
model.summary()

white_test = het_white(model.resid, model.model.exog)

labels = ["Test Statistic", "Test Statistic p-value", "F-Statistic", "F-Test p-value"]

print(dict(zip(labels, white_test)))
print("Null hypothesis - errors are homoscedastic")
print(f"significance_level({significance_level}) <= calculated p-value({white_test[1]}), null hypothesis is fail to reject") if significance_level <= white_test[1] else print(f"significance_level({significance_level}) > calculated p-value({white_test[1]}), null hypothesis is rejected")  


pd.DataFrame(white_test, index=labels).to_excel(f"white_test_{df_name.replace('.xlsx', '')}.xlsx", index=True)


# test
# from statsmodels.tools.validation import (array_like, int_like, bool_like,
#                                           string_like, dict_like, float_like)

# y = array_like(model.resid, "resid", ndim=2, shape=(x.shape[0], 1))
# x = array_like(model.model.exog, "exog", ndim=2)
# nobs, nvars0 = x.shape
# i0, i1 = np.triu_indices(nvars0)
# exog = x[:, i0] * x[:, i1]
# resols = sm.OLS(y**2, exog).fit()
# resols.summary()
# pd.DataFrame(exog).to_excel("exog_model2021.xlsx")

