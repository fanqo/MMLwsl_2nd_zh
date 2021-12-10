from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['The dog ate a sandwich and I ate a sandwich',
          'The wizard transfigured a sandwich'
          ]
vectorizer = TfidfVectorizer(stop_words='english')

print(vectorizer.fit_transform(corpus).todense())
print('标签： ' , vectorizer.get_feature_names())

# sandwich 在两文档中出现的总次数比 ate 多，但矩阵中对应的值要比 ate 对应的小
