# Marketing Response Model - Logistic Regression Baseline
# Author: Rafael Vasquez

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

df = pd.read_csv("marketing_campaign.csv", sep=";")
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
df["CustomerTenureDays"] = (pd.Timestamp.today().normalize() - df["Dt_Customer"]).dt.days

y = df["Response"].astype(int)
num_cols = ["Income","Kidhome","Teenhome","MntFishProducts","MntMeatProducts","MntFruits","MntSweetProducts","MntWines","MntGoldProds","NumDealsPurchases","NumCatalogPurchases","NumStorePurchases","NumWebPurchases","NumWebVisitsMonth","Recency","CustomerTenureDays","Year_Birth"]
cat_cols = ["Education","Marital_Status","Complain","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]

for c in num_cols: df[c] = df[c].fillna(df[c].median())
for c in cat_cols: df[c] = df[c].astype(str)

X = df[num_cols + cat_cols]
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)])

model = LogisticRegression(max_iter=200)
clf = Pipeline([("preprocess", preprocess), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

with open("marketing_metrics.txt", "w") as f:
    f.write(f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\n")
