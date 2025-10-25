#Apply Boosting Algorithm on Adult Census Income Dataset and analyze the performance of the model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# Column names
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
 "hours-per-week", "native-country", "income"]
# Read data
data = pd.read_csv(url, header=None, names=columns, na_values=" ?", 
skipinitialspace=True)
# Drop missing values
data = data.dropna()
# Features and target
X = data.drop("income", axis=1)
y = data["income"].apply(lambda x: 1 if x == ">50K" else 0) # encode target
# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='median')),
 ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='most_frequent')),
 ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
 transformers=[
 ('num', numeric_transformer, numerical_cols),
 ('cat', categorical_transformer, categorical_cols)
 ])
# Define AdaBoost with decision tree stumps
base_estimator = DecisionTreeClassifier(max_depth=1)
ada_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, 
learning_rate=1.0, random_state=42)
# Build pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
 ('classifier', ada_model)])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
stratify=y)
# Train model
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
# Evaluate
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))