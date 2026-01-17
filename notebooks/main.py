from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression  # try RandomForest later

DATA_DIR = Path("/Users/apple/Desktop/kaggle/tatanic/data")

# 1) load
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
pd.set_option('display.max_columns', 12) 
# print(train.shape)       # rows & columns
# print(train.info())      # data types and nulls
# print(train.describe())  # numeric summary
print(train.head(10))      # peek at first few rows

# quick sanity checks
print("train shape:", train.shape)
print(train.isna().sum().sort_values(ascending=False).head())

# sns.countplot(x = 'Survived', data = train)
# plt.show()

# sns.histplot(train["Age"].dropna(), bins=30)
# plt.show()

# sns.histplot(data=train, x="Age", hue="Survived", bins=30, multiple="stack")
# plt.title("Age Distribution by Survival")
# plt.show()

# sns.boxplot(x="Survived", y="Age", data=train)
# plt.title("Age vs Survival")
# plt.show()

corr = train["Age"].corr(train["Survived"])
print("Correlation between Age and Survived:", round(corr, 3))

# sns.countplot(x="Sex", hue="Survived", data=train)
# plt.title("Survival Counts by Sex")
# plt.show()

print(train["Sex"].unique())
#Cleaning

median_age = train["Age"].median()
train["Age"] = train["Age"].fillna(median_age)#filling missing age val with median

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})

X = train[["Sex", "Age"]]  
y = train["Survived"]       

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 3))

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


# sns.boxplot(x="Sex", y="Age", hue="Survived", data=train)
# plt.title("Age vs Sex vs Survival")
# plt.show()

#hypothesis: would adding Pclass as thrid value make our model better?
# print(train["Pclass"].head())

X = train[["Sex", "Age","Pclass"]]  
y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy with Sex + Age + Pclass:", round(acc, 3))

from sklearn.ensemble import RandomForestClassifier
#now see if other factors may improve our model
# print(train["Parch"].head())
# print(train["SibSp"].head())
#would sibsp + parch aka familysize would help? cuz family survive together

train["Familysize"] = train["SibSp"] + train["Parch"] + 1

# print(train["Familysize"].head())
train["Fare"].fillna(train["Fare"].median(), inplace=True)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})

features = ["Sex", "Age", "Pclass", "Familysize", "Fare", "Embarked"]
X = train[features]
y = train["Survived"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
print("Accuracy:", round(accuracy_score(y_val, y_pred), 3))

