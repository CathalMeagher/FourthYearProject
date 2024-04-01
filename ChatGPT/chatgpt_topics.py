from openai import OpenAI
import os
import json
import numpy as np

key = os.environ["CHATGPT_KEY"]
client = OpenAI(api_key=key)


# df = pd.read_csv('../more_datasets/topics.csv', header=0,
#                  names=['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7', 'summary', 'text'])
# df = df.drop(['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7'], axis=1)
# df = pd.read_csv('./more_datasets/negative_amazon_topics.csv', header=0, index_col=False)
# df = pd.read_csv('./more_datasets/test.csv', header=0,
#                  names=['sentiment', 'title', 'text'])
# df = df.drop(['sentiment', 'title'], axis=1)

def get_response_from_api(text):
    return client.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        seed=10,
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": "You are a chat bot designed to extract common opinions about a TV product from reviews. Return a list of a maximum of 5 labels, which represent common opinions about the product. The opinions must be relevant to multiple reviews. If the opinion does not appear in multiple different reviews, do not include the label in your response. The labels are not allowed to be more than two words. Use professional and factual language. Each review is separated by a new line. Return the labels in the JSON form { 'positive': [], 'mixed': [], negative': []}”"},
            # {"role": "system", "content": "You are expected to extract any positive or negative features of a given product review. Label these features with a maximum of two words. Return the labels in the JSON form { 'positive': [], 'negative': []}"},
            {"role": "user", "content": text}
        ]
    )

def get_response_from_api_single(text):
    return client.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        seed=10,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are expected to extract any positive or negative features of a given "
                                          "product review. Label these features with a maximum of two words. Return "
                                          "the labels in the JSON form { 'positive': [], 'mixed': [], 'negative': []}"},
            {"role": "user", "content": text}
        ]
    )

def get_labels(text):
    output = {
        'positive': {

        },
        'mixed': {

        },
        'negative': {

        }
    }

    for i in range(1, 6):
        response = get_response_from_api(text)
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
    return output


def build_input_from_df(df, key):
    df[key] = np.random.permutation(df[key].values)

    input = ''
    for idx, val in enumerate(df[key]):
        input += f"Review {idx}: " + val + '\n'

    return input



