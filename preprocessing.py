import time


import contractions
import pandas as pd
import re

from nltk import PorterStemmer, pos_tag
from nltk.corpus import stopwords, wordnet

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from autocorrect import Speller

df = pd.read_csv('./base_data.csv')
df = df.dropna()
spell = Speller(fast=True)
def negation_handler(sentence):
    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i-1] in ['not',"n't"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
    while '' in sentence:
        sentence.remove('')
    return sentence

def handle_negation(sentence):
    return wordnet.synsets(sentence)

def preprocess(text):
    # Lowercase the text
    try:
        text = text.lower()
    except:
        print("ERROR")
    text = contractions.fix(text)
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove urls
    text = re.sub('[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub('\n', ' ', text)  # Remove new lines

    lemmatizer = PorterStemmer()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Perform lemmatisation
    lemmatized_words = [lemmatizer.stem(token) for token in tokens]

    # Negation handling
    lemmatized_words = negation_handler(lemmatized_words)

    # Remove stopwords
    filtered_tokens = [token for token in lemmatized_words if token not in stopwords.words('english')]
    filtered_tokens = [token for token in filtered_tokens if token not in [',', '.']]

    processed_text = ' '.join(filtered_tokens)
    return spell(processed_text)

# Apply the function to our dataframe
def process():
    start = time.time()
    print("Before: ", df['reviewText'][0])
    df['reviewText'] = df['reviewText'].apply(preprocess)
    print("After: ", df['reviewText'][0])
    print(time.time() - start)
    df.to_csv('lemmatized_negation.csv', sep=',', header=True, index=False)
process()