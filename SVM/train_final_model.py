import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

data = pd.read_csv('../lemmatized_negation.csv', names=['reviewText', 'overall', 'sentiment'])
vectorizer = TfidfVectorizer()

X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)

# Map
y = [x == 'positive' for x in y]
X = vectorizer.fit_transform(X)

clf = SVC(C=10, gamma=1)

clf.fit(X,y)

joblib.dump(clf, "final_model.pkl")
joblib.dump(vectorizer, "TfidVectorizer.pkl")
