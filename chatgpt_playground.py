import numpy as np
from openai import OpenAI
from openai import OpenAI
import os
import json
import numpy as np
from preprocessing import preprocess

key = os.environ["CHATGPT_KEY"]
client = OpenAI(api_key=key)

import pandas as pd

# df = pd.read_csv('./more_datasets/topics.csv', header=0,
#                  names=['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7', 'summary', 'text'])
# df = df.drop(['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7'], axis=1)
df = pd.read_csv('./more_datasets/negative_amazon_topics.csv', header=0, index_col=False)
# df = pd.read_csv('./more_datasets/test.csv', header=0,
#                  names=['sentiment', 'title', 'text'])
# df = df.drop(['sentiment', 'title'], axis=1)


df['text'] = df['text'].apply(preprocess)
df['text'] = np.random.permutation(df['text'].values)


def get_labels(text):
    return client.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        seed=10,
        temperature=0.1,
        messages=[
            {"role": "system",
             "content": "You are a chat bot designed to extract common opinions about a TV product from reviews. Return a list of a maximum of 6 labels, which represent common opinions about the product. The opinions must be relevant to multiple reviews. If the opinion does not appear in multiple different reviews, do not include the label in your response. The labels are not allowed to be more than two words.  Use professional and factual language. Each review is separated by a new line. Return the labels in the JSON form { 'positive': [], 'mixed': [], negative': []}"},
            # {"role": "system", "content": "You are expected to extract any positive or negative features of a given product review. Label these features with a maximum of two words. Return the labels in the JSON form { 'positive': [], 'negative': []}"},
            {"role": "user", "content": text}
        ]
    )


def get_summary(text):
    return client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        seed=10,
        messages=[
            {"role": "system",
             "content": "You are expected to receive a set of product reviews. Write a short paragraph that summarizes the main aspects of the reviews. The response should in the json form {'summary': 'placeholder text'} "},
            {"role": "user", "content": text}
        ]
    )


input = ''
for idx, val in enumerate(df['text']):
    input += f"Review {idx}: " + val + '\n'

# input = 'I love this apple cider for a sweet treat without any sugar or guilt. Wonderful in the fall or winter. Much better for me than a real caramel apple or pie. It is a great decaffeinated alternative to tea or coffee and kids love it. It is very sweet, but I use extra water for a larger cup. I have also opened it up and used it to cook with. The mix inside does not brew, just rehydrates, which allows me to use it at work without a Kurieg.'
output = {
    'positive': {

    },
    'mixed': {

    },
    'negative': {

    }
}
for i in range(1, 5):
    response = get_labels(input)
    y = json.loads(response.choices[0].message.content)
    for z in output.keys():
        for label in y[z]:
            output[z][label] = output[z].get(label, 0) + 1
remove = []
for category in output:
    for label in output[category].keys():
        if output[category][label] < 3:
            remove.append((category, label))
for to_remove in remove:
    del output[to_remove[0]][to_remove[1]]
print(output)
# response = get_summary(input)
# x = json.loads(response.choices[0].message.content)

# print(x)
