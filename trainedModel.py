import time

from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from joblib import dump, load
from sklearn.model_selection import train_test_split
start = time.time()
data = pd.read_csv('lemmatized.csv')

vectorizer = CountVectorizer()

X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)
model = load('trainedmodel.joblib')


X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)

# Map
y = [x == 'positive' for x in y]

X = vectorizer.fit_transform(X)

X = X.toarray()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125

)
predicted = model.predict(X_test)

predicted_train = model.predict(X_train)
accuracy = metrics.accuracy_score(y_test, predicted)
print(accuracy)
accuracy = metrics.accuracy_score(y_train, predicted_train)
print(accuracy)
end = time.time()
print(end-start)
