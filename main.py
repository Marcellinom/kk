import joblib
author = input("Enter author: ")
title = input("Enter title: ")
print("Paste your the article content. Ctrl-D to save it.") 
contents = [] 
while True: 
    try: 
        line = input() 
    except EOFError: 
        break 
    contents.append(line) 
data = author + " " + title + " " + " ".join(contents)
model = joblib.load('model.pkl')
transformer = joblib.load('transformer.pkl')
all_words = joblib.load('all_words.pkl')
bow = all_words.transform([data])
trained_id = transformer.transform(bow)
print("Unreliable" if model.predict(trained_id)[0] == 1 else "Reliable")