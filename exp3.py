# Apply Decision Tree Algorithm on Adult Census Income Dataset and analyze the performance of the model


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

# Dataset URL (UCI Adult dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Column names (from UCI documentation)
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Download dataset if not already present
file_path = "adult.csv"
if not os.path.exists(file_path):
    print("Downloading Adult dataset...")
    data = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    data.to_csv(file_path, index=False)
else:
    print("Loading existing Adult dataset...")
    data = pd.read_csv(file_path)

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features and target
X = data.drop('income', axis=1)
y = data['income']

# Convert categorical variables using one-hot encoding
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, fontsize=8)
plt.show()
