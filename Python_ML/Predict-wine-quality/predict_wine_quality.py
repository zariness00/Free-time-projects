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
print('Training Score', f1_score(y_train, y_pred_train))
print('Testing Score', f1_score(y_test, y_pred_test))

