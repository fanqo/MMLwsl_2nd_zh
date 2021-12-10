# 词袋模型  <- 包含类似单词的文档经常有相似的含义

# 语料库，包含两个文档
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
    ]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
# CountVectorizer 将文档中的字符转换为小写，并对文档进行词汇切分
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)   # 词-索引


