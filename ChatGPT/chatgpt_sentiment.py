import time
from statistics import mean
import pandas as pd
from openai import OpenAI
from openai import OpenAI
import os
import json
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import re

from MLStripper import MLStripper

key = os.environ["CHATGPT_KEY"]
client = OpenAI(api_key=key)
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def get_labels(text):
    return client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        seed=10,
        temperature=0.001,
        messages=[
            {"role": "system",
             "content": "You are a ChatBot designed to predict the positive/negative sentiment for a product review. You will be given a list of 15 reviews, separated by a new line. Return a positive or negative label for each review, in the order that it was given to you. Make sure you return a label for each review. There should always be 15 elements in the sentiment array. Review 1 should be the first label in the response. Review 2 should be the second label in the response. Return it in the JSON format: {'predictions': [{positive|negative}]}"},
            # "content": "You are a ChatBot designed to predict the positive/negative sentiment for a product review. Analyze the whole review and return the sentiment of the review (positive/negative) in the following JSON form: {'sentiment': {POSITIVE|NEGATIVE}}"},
    {"role": "user", "content": text}
        ]
    )

def preprocess(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove urls
    text = re.sub('\n', ' ', text)  # Remove new lines
    text = strip_tags(text)
    return text



time_taken = []
scores = []

def single_review_approach():
    for _ in range(0, 5):
        data = pd.read_csv('../base_data.csv')
        data = data.sample(frac=1)

        reviews = data['reviewText'].values.astype(str)
        sentiment = data['sentiment'].values.astype(str)
        reviews = [preprocess(review) for review in reviews]
        predictions = []

        for idx, review in enumerate(reviews[0:50]):
            t = time.time()
            response = get_labels(review)
            y = json.loads(response.choices[0].message.content)
            if "sentiment" not in y:
                print("ERROR")
            else:
                predictions.append(y["sentiment"].lower())
                time_taken.append(time.time() - t)
                print(f"Request {idx} complete")
        score = accuracy_score(predictions, sentiment[0:len(predictions)])
        scores.append(score)
    print(mean(time_taken))
    print(max(time_taken))
    print(scores)
    print(mean(scores))

def chunk_approach():
    tt = time.time()
    data = pd.read_csv('../base_data.csv').sample(frac=1)
    all_predictions = []
    reviews = data['reviewText'].values.astype(str)
    sentiment = data['sentiment'].values.astype(str)
    reviews = [preprocess(review) for review in reviews]
    timings = []
    for x in range(0, 100):
        input = ""
        start = x*15
        for idx, val in enumerate(reviews[start:start+15]):
            if idx == 15:
                break
            input += f"Review {idx + 1}: " + val + "\n"
        start_time = time.time()
        response = get_labels(input)
        timings.append(time.time()-start_time)
        y = json.loads(response.choices[0].message.content)
        predictions = [x.lower() for x in y['predictions']]
        print(accuracy_score(predictions, sentiment[start:start+len(predictions)]))
        all_predictions = all_predictions + predictions
    print(time.time()-tt)
    print(mean(timings))
    print(max(timings))
    print("Accuracy score: ", accuracy_score(all_predictions, sentiment[0:len(all_predictions)]))
    confusion_matrix1 = confusion_matrix(all_predictions, sentiment[0:len(all_predictions)])
    display = ConfusionMatrixDisplay(confusion_matrix1).plot()
    plt.show()

chunk_approach()
