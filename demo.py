import json

import joblib
from data_processing import preprocessing
from ChatGPT.chatgpt_topics import get_response_from_api_single

print("Enter a review to analyse: ")
user_input = [preprocessing.preprocess(input())]

vectorizer = joblib.load("./SVM/TfidVectorizer.pkl")

input = vectorizer.transform(user_input)
# Load our model
clf = joblib.load("./SVM/final_model.pkl")

predictions = clf.predict(input)
sentiment = "positive" if predictions[0] else "negative"

response = get_response_from_api_single(user_input[0])
y = json.loads(response.choices[0].message.content)

print("This review is " + sentiment + ". The following opinions were extracted from the review: ")
for z in y.keys():
    print(z.capitalize() + ":")
    for label in y[z]:
        print(label)
    print('\n')