import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.head())


print("Dimensions :", df.shape)
print("\nColonnes :", df.columns.tolist())

print("\nTypes de données :")
print(df.dtypes)

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nRépartition des départs (Attrition) :")
print(df['Attrition'].value_counts())


cols_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
df.drop(columns=cols_to_drop, inplace=True)

print("\nNouvelles dimensions :", df.shape)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print("\nAttrition (après encodage) :")
print(df['Attrition'].value_counts())


cat_cols = df.select_dtypes(include='object').columns
print("Colonnes catégorielles :", cat_cols.tolist())


df = pd.get_dummies(df, drop_first=True)
print("Nouvelle forme du dataset :", df.shape)

#Creation of ML model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix :")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report :")
print(classification_report(y_test, y_pred))

