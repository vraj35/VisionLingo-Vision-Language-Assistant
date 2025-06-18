from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple mock data
data = {
    "cat": "A domestic animal known for its agility and sharp claws.",
    "dog": "A loyal pet animal that barks.",
    "car": "A road vehicle used for transportation.",
    "tree": "A tall plant with leaves, provides oxygen.",
    "book": "An object used for reading and gaining knowledge."
}

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data.keys())
y = list(data.values())
model = MultinomialNB()
model.fit(X, range(len(y)))

def generate_description(label):
    x = vectorizer.transform([label])
    idx = model.predict(x)[0]
    return y[idx]
