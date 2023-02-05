from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df_model = df[['mark_ychange', 'marktwert', 'typ', 'ms-name', 'gestehungskosten', 'av_mv', 'grossraum']].dropna().reset_index()
y = df_model['mark_ychange']
X = df_model[['marktwert', 'typ', 'ms-name', 'gestehungskosten', 'av_mv', 'grossraum']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Identify the columns containing categorical variables
cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Use OneHotEncoder to encode the categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train[cat_cols])

# Transform the training and test data
X_train_encoded = encoder.transform(X_train[cat_cols])
X_test_encoded = encoder.transform(X_test[cat_cols])

# Concatenate the encoded categorical data with the rest of the features
cat_col_names = [col + "_" + str(ix) for ix, col in enumerate(encoder.get_feature_names(cat_cols))]
X_train = pd.concat([X_train.drop(cat_cols, axis=1).reset_index(drop=True), pd.DataFrame(X_train_encoded.toarray(), columns=cat_col_names)], axis=1)
X_test = pd.concat([X_test.drop(cat_cols, axis=1).reset_index(drop=True), pd.DataFrame(X_test_encoded.toarray(), columns=cat_col_names)], axis=1)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)


# Calculate the accuracy score
y_test = np.where(y_test=='yield down', 0, 1)
y_pred = np.where(y_pred=='yield down', 0, 1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

importances = clf.feature_importances_

# Sort the feature importances and get the indices of the sorted features
sorted_idx = importances.argsort()[::-1]

# Print the feature names and their importances
for idx in sorted_idx:
    print(f"{X.columns[idx]}: {importances[idx]}")
    
# Calculate the AUC score
y_pred_prob = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label="ROC")
plt.plot([0, 1], [0, 1], linestyle='--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
