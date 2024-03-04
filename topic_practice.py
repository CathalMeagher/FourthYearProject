import os

import nltk
import openai
import pandas as pd
import tiktoken
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from nltk.tokenize import sent_tokenize
from bertopic.representation import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

key = os.environ["CHATGPT_KEY"]
client = openai.Client(api_key=key)

data = pd.read_csv('./more_datasets/negative_amazon_topics.csv', index_col=False)
# data = data.drop(['na1','na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7'], axis=1)

reviews = data['text']
sentences = [sent_tokenize(str(sentence)) for sentence in reviews]
sentences = [sentence for doc in sentences for sentence in doc]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Dimensionality reduction
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)

# Controlling number of topics
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Customizing count vetorizer
vectorizer_model = StemmedCountVectorizer(analyzer="word",stop_words="english", ngram_range=(1, 2))


ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
# Tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Clustering model: See [2] for more details
cluster_model = HDBSCAN(min_cluster_size = 15,
                            metric = 'euclidean',
                            cluster_selection_method = 'eom',
                            prediction_data = True)

representation_model = OpenAI(
    client,
    model="gpt-3.5-turbo",
    delay_in_seconds=2,
    chat=True,
    nr_docs=6,
    tokenizer=tokenizer
)

# topic_model = BERTopic(nr_topics='auto', representation_model=representation_model)
# topic_model = BERTopic(representation_model=representation_model, nr_topics='auto', vectorizer_model=vectorizer_model)
topic_model = BERTopic(embedding_model=embedding_model,ctfidf_model=ctfidf_model, hdbscan_model=cluster_model, vectorizer_model=vectorizer_model, representation_model=representation_model)
topics, probabilities = topic_model.fit_transform(sentences)
new_topics = topic_model.reduce_outliers(sentences, topics, strategy="c-tf-idf")
topic_model.update_topics(sentences, topics=new_topics, vectorizer_model=vectorizer_model, representation_model=representation_model)

topic_model.visualize_topics().show()
topic_model.visualize_barchart().show()
print("..")