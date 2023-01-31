from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


data = pd.read_csv("your_data.csv")
X = data.drop("target_variable", axis=1)
y = data["target_variable"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pca = PCA()
X_train_pca = pca.fit_transform(X_train)


print(pca.explained_variance_ratio_)


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

reg = LinearRegression().fit(X_train_pca, y_train)


score = reg.score(X_test_pca,y_test)
print(score)



# 2D vizualization
import matplotlib.pyplot as plt

# Fit PCA to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot of the first two principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# with lines
import matplotlib.pyplot as plt

# Fit PCA to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot of the first two principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Get the eigenvectors from PCA model
eigenvectors = pca.components_

# Scale the eigenvectors to make them more visible
eigenvectors = eigenvectors * np.sqrt(pca.explained_variance_)

# Plot the lines of direction of the original features
plt.quiver(np.mean(X_pca[:, 0]), np.mean(X_pca[:, 1]), eigenvectors[0, 0], eigenvectors[0, 1], color='r', scale=np.max(eigenvectors))
plt.quiver(np.mean(X_pca[:, 0]), np.mean(X_pca[:, 1]), eigenvectors[1, 0], eigenvectors[1, 1], color='r', scale=np.max(eigenvectors))
plt.show()

#3D vizualization

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Fit PCA to the data
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a 3D scatter plot of the first three principal components
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')

# Get the eigenvectors from PCA model
eigenvectors = pca.components_

# Scale the eigenvectors to make them more visible
eigenvectors = eigenvectors * np.sqrt(pca.explained_variance_)
ax.quiver(np.mean(X_pca[:, 0]), np.mean(X_pca[:, 1]), np.mean(X_pca[:, 2]), eigenvectors[0, 0], eigenvectors[0, 1], eigenvectors[0, 2], color='r', length=np.max(eigenvectors), arrow_length_ratio=0)
ax.quiver(np.mean(X_pca[:, 0]), np.mean(X_pca[:, 1]), np.mean(X_pca[:, 2]), eigenvectors[1, 0], eigenvectors[1, 1], eigenvectors[1, 2], color='r', length=np.max(eigenvectors), arrow_length_ratio=0)
ax.quiver(np.mean(X_pca[:, 0]), np.mean(X_pca[:, 1]), np.mean(X_pca[:, 2]), eigenvectors[2, 0], eigenvectors[2, 1], eigenvectors[2, 2], color='r', length=np.max(eigenvectors), arrow_length_ratio=0)
plt.show()

