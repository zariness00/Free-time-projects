import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

raisins = pd.read_csv('/Users/zoryawka/Desktop/Coding/Free-time-projects/Python_ML/Raisins-classification/dataset/Raisin_Dataset.csv')
print(raisins.head())

raisins["Class"] = raisins["Class"].map({"Kecimen": 0, "Besni": 1})

#Identify predictors and target 
X = raisins.drop(columns = ["Class"])
y = raisins["Class"]

print("Number of features", len(X.columns))
print("Total number of sample", len(y))
print("Number of samples belonging to Besni(1)", len(y[y == 1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
tree = DecisionTreeClassifier(random_state=42)
parameters = {
    "max_depth": [3,5,7],
    "min_samples_split": [2,3,4]
}


grid = GridSearchCV(tree, parameters)
grid.fit(X_train, y_train)
print("Best paraameters", grid.best_estimator_)
print("Best score", grid.best_score_)
print("Test score", grid.score(X_test, y_test))



df = pd.concat([pd.DataFrame(grid.cv_results_['params']), pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['Score'])], axis=1)
print(df)

# Random Search with Logistic Regression

lr = LogisticRegression(solver = 'liblinear', max_iter =1000)
distributions  = {
    "penalty": ["l1", "l2"],
    "C": uniform(loc = 0, scale = 100)

}

clf = RandomizedSearchCV(lr, distributions, n_iter = 8, random_state = 42)
clf.fit(X_train, y_train)
print("Best paraameters", clf.best_estimator_)
print("Best score", clf.best_score_)
# Print a table summarizing the results of RandomSearchCV
df = pd.concat([pd.DataFrame(clf.cv_results_['params']), pd.DataFrame(clf.cv_results_['mean_test_score'], columns=['Accuracy'])] ,axis=1)
print(df.sort_values('Accuracy', ascending = False))