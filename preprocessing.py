import pandas as pd
import re

from nltk import PorterStemmer
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from autocorrect import Speller

df = pd.read_csv('base_data.csv')

spell = Speller(fast=True)

print(len(stopwords.words("english")))
def preprocess(text):
    # Lowercase the text
    text = text.lower()

    text = re.sub('[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URls
    text = re.sub('\n', ' ', text)  # Remove new lines

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Perform lemmatisation
    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = ' '.join(lemmatized_words)
    return spell(processed_text)

# Apply the function to our dataframe
# print("Before: ", df['reviewText'][0])
# df['reviewText'] = df['reviewText'].apply(preprocess)
# print("After: ", df['reviewText'][0])
# df.to_csv('lemmatized.csv', sep=',', header=True, index=False)