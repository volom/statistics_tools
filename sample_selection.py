"""
Sample selection refers to the process of choosing a subset of data points from a larger dataset for analysis. 
There are several methods of sample selection, including:

1. Simple random sampling: In this method, each data point has an equal chance of being selected in the sample.
This method is suitable for large datasets where the population is well-defined.

2. Systematic sampling: In this method, every kth data point is selected from the population,
where k is the sample size divided by the population size. This method is suitable when the population 
is ordered and the sample size is relatively small compared to the population size.

3. Stratified sampling: In this method, the population is divided into homogeneous subgroups,
called strata, and a sample is selected from each stratum. This method is used when the population
is heterogeneous and the goal is to ensure that the sample is representative of the population.

4. Cluster sampling: In this method, the population is divided into clusters, and a sample is selected 
from each cluster. This method is used when the population is heterogeneous and the data points are 
naturally grouped into clusters.

5. Multi-stage sampling: In this method, the population is sampled in multiple stages, each of which is a sample from the previous stage. This method is used when the population is very large and it is not feasible to sample the entire population in a single stage.
"""
######################################################################################################

# Systematic sampling


import random
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Calculate the sample size and the sampling interval
population_size = df.shape[0]
sample_size = int(population_size * 0.1)
sampling_interval = int(population_size / sample_size)

# Select the first data point randomly
start = random.randint(0, sampling_interval-1)

# Create a list of indices for the systematic sample
indices = [i for i in range(start, population_size, sampling_interval)]

# Extract the systematic sample from the DataFrame
sample = df.iloc[indices, :]

######################################################################################################

# Stratified sampling



import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Divide the data into strata based on a categorical variable
strata = df.groupby('category')

# Calculate the sample size for each stratum
stratum_sizes = strata.size()
sample_size = int(df.shape[0] * 0.1)
stratum_sample_sizes = (sample_size / stratum_sizes) * stratum_sizes

# Select a sample from each stratum
samples = [group.sample(n=int(size)) for group, size in zip(strata, stratum_sample_sizes)]

# Combine the samples into a single DataFrame
stratified_sample = pd.concat(samples)

######################################################################################################

# Cluster sampling
import random
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Calculate the sample size for each stratum
stratum_sizes = df.groupby('category').size()
sample_size = int(df.shape[0] * 0.1)
stratum_sample_sizes = (sample_size / stratum_sizes) * stratum_sizes

# Select a sample from each stratum
samples = [group.sample(n=int(size)) for group, size in zip(df.groupby('category'), stratum_sample_sizes)]

# Create a new categorical variable that represents the cluster membership of each observation
cluster_labels = []
for i, data in enumerate(samples):
    labels = [i] * data.shape[0]
    cluster_labels.extend(labels)
df['cluster'] = cluster_labels

# Divide the sample into clusters based on the new categorical variable
clusters = df.groupby('cluster')

# Select a random sample of clusters
cluster_sample = random.sample(list(clusters), int(clusters.ngroups * 0.1))

# Extract the data for the selected clusters
cluster_data = [group for name, group in clusters if name in cluster_sample]

# Combine the data for the selected clusters into a single DataFrame
cluster_sample_data = pd.concat(cluster_data)


######################################################################################################

# Multi-stage sampling


import random
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Calculate the sample size for each stratum
stratum_sizes = df.groupby('category').size()
sample_size = int(df.shape[0] * 0.1)
stratum_sample_sizes = (sample_size / stratum_sizes) * stratum_sizes

# Select a sample from each stratum
samples = [group.sample(n=int(size)) for group, size in zip(df.groupby('category'), stratum_sample_sizes)]

# Create a new categorical variable that represents the cluster membership of each observation
cluster_labels = []
for i, data in enumerate(samples):
    labels = [i] * data.shape[0]
    cluster_labels.extend(labels)
df['cluster'] = cluster_labels

# Divide the sample into clusters based on the new categorical variable
clusters = df.groupby('cluster')

# Select a random sample of clusters
cluster_sample = random.sample(list(clusters), int(clusters.ngroups * 0.1))

# Extract the data for the selected clusters
cluster_data = [group for name, group in clusters if name in cluster_sample]

# Combine the data for the selected clusters into a single DataFrame
cluster_sample_data = pd.concat(cluster_data)
