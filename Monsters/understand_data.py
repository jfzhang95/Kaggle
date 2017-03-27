#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print train['type'].describe()
# sns.pairplot(train.drop('id', axis=1), hue='type', diag_kind='kde')
# plt.show()


poly_features = PolynomialFeatures(interaction_only=True)

try_comb = pd.DataFrame(
    poly_features.fit_transform(train.drop(['id', 'type', 'color'], axis=1))[:,5:],
    columns=["boneXrotting", "boneXhair", "boneXsoul",
             "rottingXhair", "rottingXsoul",
             "hairXsoul"]
)
try_comb["type"] = train.type
# sns.pairplot(try_comb, hue='type', diag_kind='kde')
# plt.show()

for i in ["boneXhair", "boneXsoul", "hairXsoul"]:
    train[i] = try_comb[i].copy()
try_comb = None

try_comb = pd.DataFrame(
    poly_features.fit_transform(test.drop(["id", "color"], axis=1))[:, 5:],
    columns=["boneXrotting", "boneXhair", "boneXsoul",
             "rottingXhair", "rottingXsoul",
             "hairXsoul"]
)


for i in ["boneXhair", "boneXsoul", "hairXsoul"]:
    test[i] = try_comb[i].copy()

try_comb = None

