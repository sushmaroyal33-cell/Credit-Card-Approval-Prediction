import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Read Dataset
# -------------------------------
data = pd.read_csv("dataset.csv")

# -------------------------------
# Handling Missing Values
# -------------------------------
data = data.fillna(0)
# -------------------------------
# Data Cleaning and Merging
# -------------------------------

print("\nData Cleaning and Merging")



# Convert all numeric columns to numeric datatype
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Fill missing values again after conversion
data = data.fillna(0)

print("Dataset cleaned successfully")
print("\nCleaned Dataset:")
print(data)

print("\nMissing Values Count")
print(data.isnull().sum())

print("\nMissing Values Percentage")
print(data.isnull().mean())

# -------------------------------
# Remove Duplicate Records
# -------------------------------
print("\nBefore removing duplicates:", len(data))

data = data.drop_duplicates()

print("After removing duplicates:", len(data))

# -------------------------------
# Univariate Analysis
# -------------------------------
print("\nApproval Status Count")
print(data["ApprovalStatus"].value_counts())

sns.countplot(x="ApprovalStatus", data=data)
plt.title("Approval Status Count")
plt.show(block=False)
plt.pause(3)
plt.close()

# -------------------------------
# Multivariate Analysis
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show(block=False)
plt.pause(3)
plt.close()

# -------------------------------
# Prepare Data
# -------------------------------
X = data.drop("ApprovalStatus", axis=1)
X = pd.get_dummies(X)

y = data["ApprovalStatus"]
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nModel trained successfully")
print("Reached Logistic Regression Step")

print("\nPredictions:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------------
# Random Forest Model
# -------------------------------

print("\n==============================")
print("Random Forest Model")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Training Random Forest Model...")

rf_pred = rf_model.predict(X_test)

print("\nPredictions:")
print(rf_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\n==============================")
print("Decision Tree Model")
print("==============================")

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

print("Training Decision Tree Model...")

dt_pred = dt_model.predict(X_test)

print("\nPredictions:")
print(dt_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, dt_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))
# -------------------------------
# Descriptive Analysis
# -------------------------------
print("\nDescriptive Statistics")
print(data.describe())

print("\nDataset Information")
data.info()