import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/zoryawka/Desktop/Coding/Free-time-projects/Python_ML/Predict-wine-quality/dataset/winequality-red.csv', sep=';')

df["quality"] = (df["quality"] > 5).astype(int)
# print(df.head(20))
# #print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(features)
X = scaler.transform(features)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
from sklearn.linear_model import LogisticRegression

clf_no_reg = LogisticRegression(penalty = None)
clf_no_reg.fit(X_train, y_train)

#Plot the coefficients obtained from the model wo regularization
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
coef.plot(kind='bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()

#For classifiers, it is important that the classifier not only has high accuracy, but also high precision and recall, i.e., a low false positive and false negative rate.
#f1 score is the weighted mean of precision and recall, captures the performance of a classifier holistically. 
from sklearn.metrics import f1_score
y_pred_train = clf_no_reg.predict(X_train)
y_pred_test = clf_no_reg.predict(X_test)    
print('Training Score no regularization', f1_score(y_train, y_pred_train))
print('Testing Score no regularization', f1_score(y_test, y_pred_test))

#Logististic regression with L2 regularization

clf_default = LogisticRegression()
clf_default.fit(X_train, y_train)
y_pred_train = clf_default.predict(X_train)
y_pred_test = clf_default.predict(X_test)    
print('Training Score L2 default', f1_score(y_train, y_pred_train))
print('Testing Score L2 default', f1_score(y_test, y_pred_test))

"""
Scores for f1 are same with no regularization and L2 regularization
Need to tune C parameter to see if it improves the score
Smaller C --> more regularization
"""

#coarse tuning of C parameter
C_array = [0.0001, 0.001, 0.01, 0.1, 1]
training_array = []
testing_array = []
for i in C_array:
    clf = LogisticRegression(C = i)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)    
    training_array.append(f1_score(y_train, y_pred_train))
    testing_array.append(f1_score(y_test, y_pred_test))
plt.plot(C_array,training_array)
plt.plot(C_array,testing_array)
plt.xscale('log')
plt.show()
plt.clf()

"""
The conclusion: The optimal C seems to be somewhere around 0.001 
"""

