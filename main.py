import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
training_data = pd.read_csv('lemmatized.csv')
test_data = pd.read_csv('lemmatized_test.csv')
test_data = test_data.sample(frac=1)

X_train = training_data['reviewText'].values.astype(str)
y_train = training_data['sentiment'].values.astype(str)

X_test = test_data['reviewText'].values.astype(str)
y_test = test_data['sentiment'].values.astype(str)

vectorizer = joblib.load("TfidVectorizer.pkl")

# Map positive/negative to True/False
y_train = [x == 'positive' for x in y_train]
y_test = [x == 'positive' for x in y_test]

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

clf = joblib.load("model.pkl")
# clf.fit(X_train, y_train)
# joblib.dump(clf, "model.pkl")

values = []
x_values = []
for i in range(10, 500, 10):
    predictions = clf.predict(X_test[:i])
    score = accuracy_score(predictions, y_test[:i])
    values.append(score)
    x_values.append(i)
    print(score)

# cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                              display_labels=clf.classes_)
# disp.plot()
# plt.show()

plt.plot(x_values, values, marker='o')
plt.xlabel("Number of reviews")
plt.ylabel("Accuracy")
plt.title("Machine Learning Accuracy Performance")
plt.ylim([0, 1])
plt.show()
