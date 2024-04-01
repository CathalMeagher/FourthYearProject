import time
from statistics import mean

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC

data = pd.read_csv('../lemmatized_negation.csv', names=['reviewText', 'overall', 'sentiment'])

vectorizer = joblib.load("../TfidVectorizer.pkl")

X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)

# Map
y = [x == 'positive' for x in y]
X = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
                        X,y,test_size = 0.30, random_state = 101)

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly']}
clf = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1)
clf.fit(X_train, y_train)
# print best parameter after tuning
print(clf.best_params_)
