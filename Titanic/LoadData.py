#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
"""


import pandas as pd
from sklearn.feature_extraction import DictVectorizer



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print train.info()
# print test.info()

selected_features = ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked', 'Fare']

X_train = train[selected_features]
X_test = test[selected_features]

Y_train = train['Survived']


# Fill Value of Embarked

X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)


# print X_train.info()
# print X_test.info()

# 特征向量化
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))
# print dict_vec.feature_names_
# print X_train[0]

