#Analyze the Boston Housing dataset and apply appropriate Regression Technique

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load Boston Housing dataset manually from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22,
header=None)
# Process raw data to extract features and target
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,
:2]])
target = raw_df.values[1::2, 2]
# Column names as per the original dataset
column_names = [
 "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
 "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
# Create DataFrame
df = pd.DataFrame(data, columns=column_names)

df['MEDV'] = target
# Display first few rows
print(df.head())
# Exploratory Data Analysis (EDA)
print(df.describe())
# Check correlations with target variable
corr_matrix = df.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))
# Plot heatmap for correlations
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation matrix of Boston Housing dataset')
plt.show()
# Feature Selection based on correlation with MEDV (target)
selected_features = ['RM', 'LSTAT', 'PTRATIO', 'DIS', 'TAX']
X = df[selected_features]
y = df['MEDV']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)

# Initialize Linear Regression model
lr = LinearRegression()
# Train the model
lr.fit(X_train, y_train)
# Predict on test data
y_pred = lr.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R2) Score: {r2:.3f}")
# Plotting Actual vs Predicted values with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted House Prices with Regression Line')
plt.show()
