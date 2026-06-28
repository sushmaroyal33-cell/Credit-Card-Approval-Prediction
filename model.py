import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dataset.csv")
import matplotlib.pyplot as plt
import seaborn as sns

print(data["ApprovalStatus"].value_counts())

sns.countplot(x="ApprovalStatus", data=data)
plt.show()
# Multivariate Analysis
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

data = data.fillna(0)

X = data.drop("ApprovalStatus", axis=1)
X = pd.get_dummies(X)

y = data["ApprovalStatus"]

model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully")
# Descriptive Analysis

print("Descriptive Statistics")
print(data.describe())

print("\nDataset Information")
print(data.info())