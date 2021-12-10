# 词袋模型  <- 包含类似单词的文档经常有相似的含义

# 语料库，包含两个文档
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
    ]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

corpus.append('I ate a sandwich')
X = vectorizer.fit_transform(corpus).todense()

# euclidean_distances 计算两个或多个向量间的距离
from sklearn.metrics.pairwise import euclidean_distances
print('第1、2文档间的距离：', euclidean_distances(X[0], X[1]))
print('第1、3文档间的距离：', euclidean_distances(X[0], X[2]))
print('第2、3文档间的距离：', euclidean_distances(X[1], X[2]))
print(euclidean_distances(X))


