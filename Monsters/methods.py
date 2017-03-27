#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   
"""

from process_data import X_train, y_train, X_test, df, df_test, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier






knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(df_test)
print(classification_report(y_pred,y_test))
Y = pd.DataFrame()

Y["id"] = indexes_test
Y["type"] = y_pred
Y.to_csv("KNN_submission.csv",index=False)




# lr = LogisticRegression(penalty='l2',C=1000000)
# lr.fit(X_train, y_train)
# y_pred = lr.predict(df_test)

# Y = pd.DataFrame()
# Y["id"] = indexes_test
# Y["type"] = y_pred
# Y.to_csv("LR_submission.csv",index=False)



# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(df_test)
# y_pred = rf.predict(df_test)
#
# Y = pd.DataFrame()
# Y["id"] = indexes_test
# Y["type"] = y_pred
# Y.to_csv("RF_submission.csv",index=False)


