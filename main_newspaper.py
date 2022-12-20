from newspaper import Article
import joblib
url = 'https://www.nytimes.com/2022/12/15/world/europe/qatar-european-parliament-bribery.html'
article = Article(url)
article.download()
article.parse()
if article is None or article.authors[0] is None or article.title is None or article.text is None:
    print("error in parsing")
else:
    author = article.authors[0]
    title = article.title
    contents = article.text
    data = author + " " + title + " " + " ".join(contents)
    model = joblib.load('model.pkl')
    transformer = joblib.load('transformer.pkl')
    all_words = joblib.load('all_words.pkl')
    bow = all_words.transform([data])
    trained_id = transformer.transform(bow)
    print("Unreliable" if model.predict(trained_id)[0] == 1 else "Reliable")
