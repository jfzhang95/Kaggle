#!usr/bin/env python
#-*- coding:utf-8 -*-

"""
@author: James Zhang
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from logRegres import *
from LoadData import X_train, X_test, Y_train, test
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
import csv

# logistic regression
def logregression(X_train, X_test, Y_train):
    X = []
    Y = []

    for ele in X_train:
        X.append(ele)
    for ele in Y_train:
        Y.append(ele)
    X = np.array(X)
    Y =  np.array(Y)
    # w = logregres_train(X, Y)
    # print w

    total_acc = 0.0
    iters = 10
    for i in xrange(iters):
        weights = logregres_train(X, Y)
        acc = logregres_test(X, Y, weights)
        total_acc += np.float(acc)

    print "mean_acc:", total_acc / iters, "%"

    X_test = np.array(X_test)

    Y_pred = logregres_predict(X_test, weights)

    return Y_pred

# logreg run
PRED = logregression(X_train, X_test, Y_train)
# with open('submission.csv', 'wb') as MyFile:
#     myWriter = csv.writer(MyFile)
#     myWriter.writerow(["PassengerId", "Survived"])
#     index = 0
#     for i in PRED:
#         tmp = []
#         tmp.append(test['PassengerId'][index])
#         tmp.append(int(i))
#         myWriter.writerow(tmp)
#         index += 1



def RandomForest(X_train, Y_train, X_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    print cross_val_score(rfc, X_train, Y_train, cv=5).mean()
    Y_pred = rfc.predict(X_test)
    return Y_pred

# RF run
# PRED = RandomForest(X_train, Y_train, X_test)



def XGBClissifier():
    pass
