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
