from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))


# 需要先使用 nltk.download('wordnet') 下载 WordNet 资源

