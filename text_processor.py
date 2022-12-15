import string
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sys
stopword = stopwords.words('english')

def process_text(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopword]

def process_bow(df):
    all_words = CountVectorizer(analyzer=process_text).fit(df['content'].values.astype('U'))
    return all_words.transform(df['content'].values.astype('U'))

def id_bow(bow):
    tfidf_transformer = TfidfTransformer().fit(bow)
    return tfidf_transformer.transform(bow)
