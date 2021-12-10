from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()

X = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
    ]

# DictVectorizer将 特征-值 列表 转换为矢量
print(onehot_encoder.fit_transform(X).toarray())
