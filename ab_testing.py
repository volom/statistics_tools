from scipy import stats

# Check normality of A group
A_group = [1, 2, 3, 4, 5]
stats.shapiro(A_group)

# Check normality of B group
B_group = [6, 7, 8, 9, 10]
stats.shapiro(B_group)

"""
The output will be a tuple containing the test statistic and 
the p-value. If the p-value is greater than 0.05, it indicates 
that the data is likely to be normally distributed.
"""

# Equal variances check:

from scipy.stats import levene

A_group = [1, 2, 3, 4, 5]
B_group = [6, 7, 8, 9, 10]
levene(A_group, B_group)
"""
The output will be a tuple containing the test statistic and the p-value.
If the p-value is greater than 0.05, it indicates that the variances 
of the two groups are likely to be equal.
"""


# chi-squared
from scipy.stats import chisquare

observed = [3, 5, 6, 7, 8]
expected = [5, 5, 5, 5, 5]

chisquare(observed, expected)


#ttest
import numpy as np
from scipy import stats

# Define the number of visitors in each group
n_A = 1000
n_B = 1000

# Define the conversion rate for the control group (A)
p_A = 0.1

# Define the conversion rate for the variation group (B)
p_B = 0.11

# Generate random samples for each group
A_conversions = np.random.binomial(n_A, p_A)
B_conversions = np.random.binomial(n_B, p_B)

# Calculate the conversion rate for each group
A_cr = A_conversions / n_A
B_cr = B_conversions / n_B

# Perform a two-sided t-test
t, p = stats.ttest_ind(A_cr, B_cr)

# Print the results
print("t-value: ", t)
print("p-value: ", p)

# Interpret the results
alpha = 0.05
if p < alpha:
    print("The difference in conversion rates is statistically significant.")
else:
    print("The difference in conversion rates is not statistically significant.")

# Bayesian approach

import pymc3 as pm

def bayesian_ab_test(A_conversions, A_visitors, B_conversions, B_visitors, significance_level):
    # Define the prior distributions for the conversion rates
    with pm.Model() as model:
        A_cr = pm.Beta('A_cr', alpha=1, beta=1)
        B_cr = pm.Beta('B_cr', alpha=1, beta=1)

    # Define the likelihood function
    with model:
        A_like = pm.Binomial('A_like', n=A_visitors, p=A_cr, observed=A_conversions)
        B_like = pm.Binomial('B_like', n=B_visitors, p=B_cr, observed=B_conversions)

    # Perform Markov Chain Monte Carlo (MCMC) sampling
    with model:
        trace = pm.sample(1000, chains=1)

    # Get the difference of means 
    diff_of_means = trace['A_cr'] - trace['B_cr']
    
    # Compute the probability that A_cr is greater than B_cr
    prob = (diff_of_means > 0).mean()
    
    # check if the probability is greater than the significance level
    if prob > significance_level:
        print(f'A version is better with a probability of {prob:.2f}')
    else:
        print(f'B version is better with a probability of {1-prob:.2f}')
    
    # plot the results
    pm.plot_posterior(trace, var_names=['A_cr', 'B_cr'], ref_val=0.0, color='#87ceeb', alpha=0.2)
    
A_conversions = 500
A_visitors = 1000
B_conversions = 550
B_visitors = 1000
significance_level = 0.05
bayesian_ab_test(A_conversions, A_visitors, B_conversions, B_visitors, significance_level)

"""
other distributions can be used in the likelihood function depending on the type of data you have. 
The binomial distribution, which is used in the example is commonly used for binary data, 
such as the number of conversions in an A/B test. However, if you have continuous data, you might use
a normal distribution or a student t-distribution in the likelihood function.
"""
with pm.Model() as model:
    A_time = pm.Normal('A_time', mu=10, sd=1)
    B_time = pm.Normal('B_time', mu=10, sd=1)
    A_like = pm.Normal('A_like', mu=A_time, sd=1, observed=A_data)
    B_like = pm.Normal('B_like', mu=B_time, sd=1, observed=B_data)

