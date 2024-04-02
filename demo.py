import json

import joblib
from data_processing import preprocessing
from ChatGPT.chatgpt_topics import get_response_from_api_single

print("Enter a review to analyse: ")
base_user_input = input()
user_input = [preprocessing.preprocess(base_user_input)]

vectorizer = joblib.load("./SVM/TfidVectorizer.pkl")

input = vectorizer.transform(user_input)
# Load our model
clf = joblib.load("./SVM/final_model.pkl")

predictions = clf.predict(input)
sentiment = "Positive" if predictions[0] else "Negative"

response = get_response_from_api_single(base_user_input)
y = json.loads(response.choices[0].message.content)

print("This review is predicted as being " + sentiment + ".\nThe following opinions were extracted from the review: \n")
for z in y.keys():
    print(z.capitalize() + ":")
    for label in y[z]:
        print(label.capitalize())
    print('\n')