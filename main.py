import time

import joblib
import numpy as np
import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import preprocessing

data = pd.read_csv('lemmatized.csv')

X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)

vectorizer = joblib.load("TfidVectorizer.pkl")

# Map positive/negative to True/False
y = [x == 'positive' for x in y]
# n_estimators = 10
X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size=0.4, random_state=0)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

clf = joblib.load("model.pkl")
# clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))
# joblib.dump(clf, "model.pkl")
# joblib.dump(vectorizer, "TfidVectorizer.pkl")

# disp = ConfusionMatrixDisplay.from_estimator(
#     clf,
#     X_test,
#     y_test,
# )
# plt.show()

# defining parameter range
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}
#
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
#
# # fitting the model for grid search
# grid.fit(X_train, y_train)
# print(grid)