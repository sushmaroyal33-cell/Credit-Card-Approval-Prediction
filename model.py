import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dataset.csv")
import matplotlib.pyplot as plt
import seaborn as sns

print(data["ApprovalStatus"].value_counts())

sns.countplot(x="ApprovalStatus", data=data)
plt.show()

data = data.fillna(0)

X = data.drop("ApprovalStatus", axis=1)
X = pd.get_dummies(X)

y = data["ApprovalStatus"]

model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully")