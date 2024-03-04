# Data processing
import pandas as pd
import numpy as np
# Text preprocessiong
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

wn = nltk.WordNetLemmatizer()
# Topic model
from bertopic import BERTopic
# Dimension reduction
from umap import UMAP


df = pd.read_csv('./more_datasets/topics.csv', header=0,
                 names=['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7', 'summary', 'text'])
df = df.drop(['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7'], axis=1)

docs = df['text']
vectorizer_model = CountVectorizer(stop_words="english")

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model_embedding.encode(docs)

model = BERTopic(
    n_gram_range=(1, 2),
    vectorizer_model=vectorizer_model,
    nr_topics='auto',
    calculate_probabilities=True).fit(docs, corpus_embeddings)

topics, probabilities = model.transform(docs, corpus_embeddings)

new_topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf")
model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
model.visualize_topics().show()
model.visualize_barchart().show()