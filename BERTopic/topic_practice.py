import os
from bertopic.representation import OpenAI
import openai
import pandas as pd
import tiktoken
from bertopic import BERTopic
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

key = os.environ["CHATGPT_KEY"]
client = openai.Client(api_key=key)

data = pd.read_csv('../more_datasets/topics.csv', header=0,
                   names=['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7', 'summary', 'text'])

# data = data.drop(['na1','na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7'], axis=1)

reviews = data['text']
sentences = [sent_tokenize(str(sentence)) for sentence in reviews]
sentences = [sentence for doc in sentences for sentence in doc]

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(sentences, show_progress_bar=True)

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

from umap import UMAP

# Reduce dimensionality and provide stochastic behaviour
umap_model = UMAP(n_neighbors=150, metric='cosine', random_state=42)


# Used to control clusters
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True, allow_single_cluster=True)

# Tokenizer
tokenizer= tiktoken.encoding_for_model("gpt-3.5-turbo")

representation_model = OpenAI(
    client,
    model="gpt-3.5-turbo",
    delay_in_seconds=2,
    chat=True,
    nr_docs=8,
    doc_length=100,
    tokenizer=tokenizer
)

topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,
  umap_model=umap_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(sentences, embeddings)

# Visualize topics with custom labels
topic_model.visualize_topics().show()
topic_model.visualize_barchart(top_n_topics=5, custom_labels=True).show()

# Reduce outliers
new_topics = topic_model.reduce_outliers(sentences, topics)
print(len(new_topics))