import pandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('landmarkdata2.csv')

X = df.drop('63', axis=1)
y = df['63']

print(y[78], y[84], y[110])

le = LabelEncoder()
y = le.fit_transform(y)

print(y[78], y[84], y[110])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ロジスティック回帰モデルで学習を行い、テストデータのスコアを表示
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

with open('./model/logistic3.pkl', 'wb') as f:
    pickle.dump(lr, f)
