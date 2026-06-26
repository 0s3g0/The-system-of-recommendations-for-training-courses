import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Обработка текста
def process_text(text):
    stemmer = PorterStemmer() # Инициализация стеммера

    # Убираем пунктуацию
    trans = str.maketrans('','', string.punctuation)
    cleaned_text = text.translate(trans)

    tokens = word_tokenize(cleaned_text)
    filtered_words = [w for w in tokens if not w.lower() in stopwords.words('english')] # Убираем слова лишенные смысловой нагрузки
    stemmed_words = [stemmer.stem(word) for word in filtered_words] # Сокращаем слова до условно "начальной формы"

    return stemmed_words
