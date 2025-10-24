#Analyze the Titanic Survival Dataset and apply appropriate regression technique


# Titanic Survival Prediction with feature importance

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Dataset URL
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
file_path = "titanic.csv"

# Download if missing
if not os.path.exists(file_path):
    print("Downloading Titanic dataset...")
    data = pd.read_csv(titanic_url)
    data.to_csv(file_path, index=False)
else:
    print("Loading existing Titanic dataset...")
    data = pd.read_csv(file_path)

# Features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Convert categorical to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Define X and y
X = data[features]
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}\n")

# Coefficients
coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Feature impact on survival (positive = more likely to survive):\n")
print(coeff_df.to_string(index=False))
