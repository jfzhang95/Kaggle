#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# sns.pairplot(df, hue='type')

Y = df['type']

indexes_test = df_test['id']
df = df.drop(['type','color','id'], axis=1)
df_test = df_test.drop(['color','id'], axis=1)

df = pd.get_dummies(df)
df_test = pd.get_dummies(df_test)


X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.3, random_state=0)

