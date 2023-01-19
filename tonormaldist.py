import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Generate some non-normal data
data = np.random.exponential(size=1000)

# Perform the Shapiro-Wilk test to check for normality
w, p = shapiro(data)
print("Shapiro-Wilk test p-value: ",p)

# Logarithmic transformation
log_data = np.log(data)
w, p = shapiro(log_data)
print("Shapiro-Wilk test p-value after log transformation: ",p)

# Square root transformation
sqrt_data = np.sqrt(data)
w, p = shapiro(sqrt_data)
print("Shapiro-Wilk test p-value after sqrt transformation: ",p)

# Box-Cox transformation
from scipy.stats import boxcox
bc_data, _ = boxcox(data)
w, p = shapiro(bc_data)
print("Shapiro-Wilk test p-value after box-cox transformation: ",p)

# Plot histograms of the original and transformed data
plt.figure(figsize=(12,5))
plt.subplot(131)
plt.hist(data)
plt.title("Original Data")

plt.subplot(132)
plt.hist(log_data)
plt.title("Log Transformed Data")

plt.subplot(133)
plt.hist(sqrt_data)
plt.title("Sqrt Transformed Data")

plt.show()
