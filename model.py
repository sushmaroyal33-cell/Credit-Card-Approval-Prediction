import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

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

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("\nModel trained successfully")

# -------------------------------
# Descriptive Analysis
# -------------------------------
print("\nDescriptive Statistics")
print(data.describe())

print("\nDataset Information")
data.info()