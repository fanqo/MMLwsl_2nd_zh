from sklearn.feature_extraction.text import HashingVectorizer

corpus = ['the', 'ate', 'bacon', 'cat']
vectorizer = HashingVectorizer(n_features = 6)
X = vectorizer.transform(corpus)
print(X.todense())
print(X.shape)
