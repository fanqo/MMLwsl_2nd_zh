# 词袋模型  <- 包含类似单词的文档经常有相似的含义

# 语料库，包含两个文档
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
    ]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

corpus.append('I ate a sandwich')
X = vectorizer.fit_transform(corpus).todense()

print(X)
print(vectorizer.vocabulary_)

from sklearn.metrics.pairwise import euclidean_distances
print(euclidean_distances(X))


