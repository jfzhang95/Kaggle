#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   
"""

# code from:
# https://www.kaggle.com/zyedpls/house-prices-advanced-regression-techniques/regularized-linear-models

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats.stats import pearsonr
import matplotlib
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))

# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
# prices.hist()

# log transform the target:
train['SalePrice'] = np.log1p(train['SalePrice'])

# log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # 计算偏度
"""
偏度与峰度类似，它也是描述数据分布形态的统计量，其描述的是某总体取值分布的对称性。
这个统计量同样需要与正态分布相比较，偏度为0表示其数据分布形态与正态分布的偏斜程度相同；
偏度大于0表示其数据分布形态与正态分布相比为正偏或右偏，数据右端有较多的极端值；
偏度小于0表示其数据分布形态与正态分布相比为负偏或左偏，数据左端有较多的极端值。
偏度的绝对值数值越大表示其分布形态的偏斜程度越大。
"""
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# one-hot
all_data = pd.get_dummies(all_data)

# filling NA's with the mean of the colum:
all_data = all_data.fillna(all_data.mean())

# create matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test  = all_data[train.shape[0]:]
Y_train = train.SalePrice
print Y_train

#
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
#
# def rmse_cv(model):
#     rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = 10))
#     return(rmse)
#
#
# model_ridge = Ridge()
# # alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
# # cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
# # cv_ridge = pd.Series(cv_ridge, index=alphas)
#
# # cv_ridge.plot(title='validation')
# # plt.xlabel('alpha')
# # plt.ylabel('rmse')
# #
# model_ridge = Ridge(alpha=5).fit(X_train, Y_train)
#
# # let's look at the residuals as well:
# # matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
# #
# # preds_ridge = pd.DataFrame({"preds Ridge":model_ridge.predict(X_train), "true":Y_train})
# # preds_ridge["residuals"] = preds_ridge["true"] - preds_ridge["preds Ridge"]
# # preds_ridge.plot(x = "preds Ridge", y = "residuals", kind = "scatter")
#
#
# model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y_train)
#
# # Another neat thing about the Lasso is that
# # it does feature selection for you
# # setting coefficients of features it deems unimportant to zero.
# # Let's take a look at the coefficients:
#
#
# # 去除相关系数为0的特征
# coef = pd.Series(model_lasso.coef_, index=X_train.columns)
# # print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#
# imp_coef = pd.concat([coef.sort_values().head(10),
#                       coef.sort_values().tail(10)])
#
# # print imp_coef
#
# # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# # imp_coef.plot(kind = "barh")
# # plt.title("Coefficients in the Lasso Model")
#
#
# #let's look at the residuals as well:
# # matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
# #
# # preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":Y_train})
# # preds["residuals"] = preds["true"] - preds["preds"]
# # preds.plot(x = "preds", y = "residuals",kind = "scatter")
# #
# # plt.show()
#
#
# model_elas = ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(X_train, Y_train)
#
# import xgboost as xgb
#
# dtrain = xgb.DMatrix(X_train, label = Y_train)
# dtest = xgb.DMatrix(X_test)
#
# params = {"max_depth":6, "eta":0.1}
# model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
#
# # model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
# # plt.show()
#
# model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1) #the params were tuned using xgb.cv
# model_xgb.fit(X_train, Y_train)
#
#
# #gb_preds = np.expm1(model_gb.predict(X_test))
# lasso_preds = np.expm1(model_lasso.predict(X_test))
# ridge_preds = np.expm1(model_ridge.predict(X_test))
# xgb_preds = np.expm1(model_xgb.predict(X_test))
# elas_preds = np.expm1(model_elas.predict(X_test))
#
# predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
# predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
#
#
#
# preds = 0.15 * xgb_preds + 0.75 * elas_preds + 0.10 * elas_preds
# solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
# solution.to_csv("ridge_sol.csv", index = False)
