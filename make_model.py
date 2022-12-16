import pandas as pd
import joblib

df = pd.read_csv('train.csv')
df = df.head(20000)

df.fillna('')
df['content'] = df['author'] + ' ' + df['title'] + ' ' + df['text']

df = df.drop(['text', 'author', 'title'],axis=1)

from text_processor import count_vectorize, id_transformer
from sklearn.model_selection import train_test_split

all_words = count_vectorize(df)
bow = all_words.transform(df['content'].values.astype('U'))

transformer = id_transformer(bow)
trained_id = transformer.transform(bow)

X_train, X_test, y_train, y_test = train_test_split(trained_id, df['label'], test_size=0.01, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

joblib.dump(lr, 'model.pkl')
joblib.dump(transformer, 'transformer.pkl')
joblib.dump(all_words, 'all_words.pkl')