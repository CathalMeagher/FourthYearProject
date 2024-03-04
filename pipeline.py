import json

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

from chatgpt_playground import get_labels, get_summary
from preprocessing import preprocess

# Load the dataset
data = pd.read_csv('./MyOwnProcessedData.csv')

data = data.drop(['overall', 'summary'], axis=1)

data = data[:2000]

# Apply preprocessing
data['reviewText'] = data['reviewText'].apply(preprocess)

# Label sentiment
X = data['reviewText'].values.astype(str)
y = data['sentiment'].values.astype(str)
y = [x == 'positive' for x in y]

vectorizer = joblib.load("TfidVectorizer.pkl")

X_train = vectorizer.transform(X)
# Load our model
clf = joblib.load("model_2.pkl")

predictions = clf.predict(X_train)

cm = confusion_matrix(y, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()
value = data['reviewText'][0:200].values[0:200]

input = ''
for idx, val in enumerate(value):
   input += f"Review {idx}: " + val

pos = 0
neg = 0
for prediction in predictions:
    if prediction:
        pos += 1
    else:
        neg += 1

print(pos)
print(neg)

response = get_labels(input)
y = json.loads(response.choices[0].message.content)
response = get_summary(input)
x = json.loads(response.choices[0].message.content)

print(y['positive'])
print(y['negative'])
print(x['summary'])