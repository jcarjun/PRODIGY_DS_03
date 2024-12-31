# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# Load data
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Display metadata and variable information
print("Dataset Metadata:\n", bank_marketing.metadata)
print("\nDataset Variables:\n", bank_marketing.variables)

# Step 1: Data Preprocessing
# Check for missing values
print("\nMissing Values in Features:\n", X.isnull().sum())
print("\nMissing Values in Targets:\n", y.isnull().sum())

# Fill missing values (if necessary) or drop rows with missing data
X.fillna(method="ffill", inplace=True)  # Replace missing values with forward fill
y.fillna(method="ffill", inplace=True)

# Convert categorical variables to numeric using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Step 2: Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Building the Decision Tree Classifier
# Initialize the classifier
dt_classifier = DecisionTreeClassifier(
    criterion="gini", max_depth=5, random_state=42
)

# Train the model
dt_classifier.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predict on the test data
y_pred = dt_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of Decision Tree Classifier:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)  # Pass y_test and y_pred
print("\nClassification Report:\n", class_report)

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt_classifier,
    feature_names=X_encoded.columns,
    class_names=[str(c) for c in dt_classifier.classes_],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree Visualization")
plt.show()

# Step 6: Insights
print("\nKey Insights:")
print("- The decision tree splits data based on features that contribute most to distinguishing between customers who purchase and those who do not.")
print("- Accuracy indicates the overall performance of the classifier.")
print("- Analyze feature importance for understanding influential variables.")
