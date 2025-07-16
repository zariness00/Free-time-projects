import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
file_path = "/Users/zoryawka/Desktop/Coding/Free-time-projects/Python_ML/Income-prediction/data/adult.data"
census_data = pd.read_csv(file_path, header=None, names = col_names)

# Distinct values in 'income' column

print(census_data['income'].value_counts(normalize=True))

#Clean columns by stripping extra whitespace for columns of type "object"

for col in census_data.columns:
    if census_data[col].dtype == 'object':
        census_data[col] = census_data[col].str.strip()


#Create feature dataframe X with feature columns and dummy variables for categorical features
feature_cols = ['age', 'workclass', 'marital-status', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex','race']
X = pd.get_dummies(census_data[feature_cols], drop_first=True)

#Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greather than 50k
y =  census_data["income"].apply(lambda x:1 if x == ">50K" else 0)

#Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Instantiate random forest classifier, fit and score with default parameters
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
classifier_score = classifier.score(X_test, y_test)
print(f'Accuracy score for default random forest: {round(classifier.score(X_test, y_test)*100,3)}%')

#Tune the hyperparameter max_depth over a range from 1-25, save scores for test and train set
np.random.seed(0)
accuracy_train=[]
accuracy_test = []

for i in range(1, 26):
    classifier = RandomForestClassifier(max_depth=i, random_state=42)
    classifier.fit(X_train, y_train)
    accuracy_train.append(classifier.score(X_train, y_train))
    accuracy_test.append(classifier.score(X_test, y_test))

#Find the best accuracy and at what depth that occurs
print('Maximum accuracy on test set:', np.max(accuracy_test))
print('Max depth for maximum accuracy on test set:', np.argmax(accuracy_test) +1)  # +np.argmax() returns a zero-based index

#Plot the accuracy scores for the test and train set over the range of depth values  
plt.figure(figsize=(10, 6))
plt.plot(range(1, 26), accuracy_train, label='Train Accuracy', marker='o')
plt.plot(range(1, 26), accuracy_test, label='Test Accuracy', marker='o')
plt.title('Random Forest Classifier Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.xticks(range(1, 26))
plt.legend()
plt.grid()
plt.show()

#Save the best random forest model and save the feature importances in a dataframe
best_classifier = RandomForestClassifier(max_depth=18, random_state=42)
best_classifier.fit(X_train, y_train)
feature_import_df = pd.DataFrame(zip(X_train.columns, best_classifier.feature_importances_),  columns=['feature', 'importance'])
print('Top 5 random forest features:')
print(feature_import_df.sort_values('importance', ascending=False).iloc[0:5])
