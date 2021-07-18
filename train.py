import os
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler


SRC_FILE = './data/6 class csv.csv'
df = pd.read_csv(SRC_FILE)
print(df.head())
excluded_columns = ['star_type']

#Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , SuperGiants, HyperGiants
label_column = df['star_type']

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numeric_features = [col for col in numeric_columns if col not in excluded_columns]

categorical_features = [col for col in categorical_columns if col not in excluded_columns]
features = numeric_features + categorical_features

df = df[features]
df = df.fillna(0)

X_train, X_valid, y_train, y_valid = train_test_split(df, label_column, 
                                                        test_size=0.2, random_state=42, stratify = label_column)

le = OrdinalEncoder(categorical_features)
le.fit(X_train[categorical_features])



X_train[categorical_features] = le.transform(X_train[categorical_features])
X_valid[categorical_features] = le.transform(X_valid[categorical_features])

scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

num_leaves = [10,20,30,40,50]
max_depth = [2,3,4,5,6,7,8]
learning_rate = [0.05, 0.01, 0.005, 0.001]
result = []

joblib.dump(le, './model/label_encoder.joblib')
joblib.dump(scaler, './model/minmax_scaler.joblib')
joblib.dump(features, './model/features.joblib')
joblib.dump(categorical_features, './model/categorical_features.joblib')



for num in num_leaves:
    for depth in max_depth:
        for lr in learning_rate:
            clf = LGBMClassifier(random_state = 420, 
                                num_leaves = num, 
                                max_depth = depth, 
                                learning_rate = lr)
            clf.fit(X_train, y_train)

            valid_prediction = clf.predict(X_valid)
            accuracy = accuracy_score(valid_prediction, y_valid)
            f1 = f1_score(valid_prediction, y_valid, average = 'weighted')

            metadata = {"num_leaves ": num,
                        "max_depth": depth,
                        "learning_rate": lr,
                        "accuracy": accuracy, 
                        "f1_score": f1}
            print(metadata)
            print(classification_report(y_valid,valid_prediction))
            result.append(metadata)

            joblib.dump(clf, f'./model/lgb_model_{f1}.joblib')

print(result)

