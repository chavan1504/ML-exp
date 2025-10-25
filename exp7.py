import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# Load dataset
column_names = [
 "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
 "hours-per-week", "native-country", "income"
]
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(data_url, header=None, names=column_names, na_values=' ?')
# Check and drop missing values
print(f"Missing values in each column:\n{data.isnull().sum()}")
data = data.dropna()
# Separate features and target
X = data.drop("income", axis=1)
y = data["income"]
# Encode target
le = LabelEncoder()
y = le.fit_transform(y) # 0 = <=50K, 1 = >50K
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Pipelines for preprocessing
numerical_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy='median')),
 ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy='most_frequent')),
 ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
 ('num', numerical_pipeline, numerical_cols),
 ('cat', categorical_pipeline, categorical_cols)
])
# Preprocess features
X_processed = preprocessor.fit_transform(X)
print(f"Shape before dimensionality reduction: {X_processed.shape}")
# Use TruncatedSVD for dimensionality reduction (suitable for sparse matrix)
n_components = 50 # Tune this number as needed
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X_processed)
print(f"Shape after TruncatedSVD: {X_reduced.shape}")
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
 X_reduced, y, test_size=0.2, random_state=42, stratify=y
)
# Train Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))