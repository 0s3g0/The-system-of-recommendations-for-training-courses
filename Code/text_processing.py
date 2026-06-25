import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def process_text(text):
    stemmer = PorterStemmer()

    trans = str.maketrans('','', string.punctuation)
    cleaned_text = text.translate(trans)

    tokens = word_tokenize(cleaned_text)
    filtered_words = [w for w in tokens if not w.lower() in stopwords.words('english')]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return stemmed_words
