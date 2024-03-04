import time

import numpy as np
import pandas as pd
from openai import OpenAI
from openai import OpenAI
import os
import json
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

key = os.environ["CHATGPT_KEY"]
client = OpenAI(api_key=key)


def get_labels(text):
    return client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        seed=10,
        messages=[
            {"role": "system",
             "content": "You are expected to predict the sentiment for a product review. You will be given a list of reviews, separated by a new line. Return a positive or negative label for each review, in the order that it was given to you. Review 1 should be the first label in the response. Review 2 should be the second label in the response. Return it in the JSON format: {'predictions': [{Positive|Negative}]}"},
            {"role": "user", "content": text}
        ]
    )


data = pd.read_csv('base_data.csv')
reviews = data['reviewText'].values.astype(str)
sentiment = data['sentiment'].values.astype(str)
scores = []
x_values = []
input = ''
for idx, val in enumerate(reviews):
    if idx == 100:
        break
    input += f"Review {idx+1}: " + val + "\n"

t = time.time()
response = get_labels(input)
print(time.time()-t)
y = json.loads(response.choices[0].message.content)
predictions = [x.lower() for x in y['predictions']]
# score = accuracy_score(predictions, sentiment[:len(predictions)])
# scores.append(score)

plt.plot(x_values, scores, marker='o')
plt.ylabel("Accuracy")
plt.xlabel("Number of reviews")
plt.title("ChatGPT Accuracy Performance")
plt.show()