import time
from statistics import mean

import joblib
from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from preprocessing import preprocess

import preprocessing

start = time.time()
data = pd.read_csv('./lemmatized_negation.csv', names=['reviewText', 'overall', 'sentiment'])

vectorizer = joblib.load("TfidVectorizer.pkl")


# {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
# data['reviewText'] = data['reviewText'].apply(preprocess)
X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)
model = SVC(C=10, gamma=1)

# Map
y = [x == 'positive' for x in y]
X = vectorizer.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33

)

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly']}
scores = cross_val_score(model, X, y, cv=5)

# results = model.predict(X_test)

print(mean(scores))