#!usr/bin/env python
#-*- coding:utf-8 -*-

"""
@author: James Zhang
"""

import numpy as np
from LoadData import X_train, X_test, Y_train, test
import csv
from methods import *


# logreg run
# logregression(X_train, X_test, Y_train)


# RF run
# RandomForest(X_train, Y_train, X_test)

# XGB run
XGBoosting(X_train, Y_train, X_test)





