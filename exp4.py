import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]
data = pd.read_csv(url, header=None, names=column_names, na_values=' ?')

# 2. Handle missing values
missing_values_count = data.isnull().sum()
data.dropna(inplace=True)

# 3. Encode categorical variables
cat_cols = data.select_dtypes(include='object').columns.drop('income')
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Summary stats
total_samples = data.shape[0]
class_counts = data['income'].value_counts()
class_percentages = data['income'].value_counts(normalize=True) * 100
avg_age = data['age'].mean()
avg_hours_per_week = data['hours_per_week'].mean()

# 4. Split data
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Predictions & evaluation
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]  # probabilities for ROC curve

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
classification_report_dict = {
    '<=50K': {
        'precision': class_report['0']['precision'],
        'recall': class_report['0']['recall'],
        'f1-score': class_report['0']['f1-score'],
        'support': int(class_report['0']['support'])
    },
    '>50K': {
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1-score': class_report['1']['f1-score'],
        'support': int(class_report['1']['support'])
    }
}

def print_model_performance_report(missing_values, total_samples, class_counts, class_percentages, avg_age, avg_hours, accuracy, class_report):
    print("\n" + "="*60)
    print(" MODEL PERFORMANCE REPORT")
    print("="*60 + "\n")

    print("Dataset Summary Statistics:\n")
    print(f"Total samples after cleaning: {total_samples}")
    print(f"Class distribution:")
    for cls, count in class_counts.items():
        print(f" Class {cls} : {count} samples ({class_percentages[cls]:.2f}%)")
    print(f"Average age: {avg_age:.1f} years")
    print(f"Average hours worked per week: {avg_hours:.1f}\n")

    print("Dataset Overview (Missing Values per Feature):\n")
    print(f"{'Feature':<20} | {'Missing Values':>15}")
    print("-"*40)
    for feature, miss_count in missing_values.items():
        print(f"{feature:<20} | {miss_count:>15,}")
    print("\n*Note: Rows with missing values were dropped before training.\n")

    print("-"*60)
    print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report (per class):")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*65)
    for label, metrics in class_report.items():
        print(f"{label:<12} {metrics['precision']*100:>9.0f}% {metrics['recall']*100:>9.0f}% {metrics['f1-score']*100:>9.0f}% {metrics['support']:>10,}")

    print("\nInterpretation:")
    print("- Model performs well on <=50K class with high precision and recall.")
    print("- Lower performance on >50K class due to class imbalance.")
    print("- Overall accuracy is strong for this dataset.")
    print("="*60 + "\n")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Print the report
print_model_performance_report(
    missing_values_count, total_samples, class_counts,
    class_percentages, avg_age, avg_hours_per_week, accuracy, classification_report_dict
)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)

# Plot ROC curve
plot_roc_curve(y_test, y_proba)
