from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'I am gathering ingredients for the sandwich.',
    'There were many wizards at the gathering.'
    ]

vectorizer = CountVectorizer(binary=True, stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
