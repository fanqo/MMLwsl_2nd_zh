from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

wordnet_tags = ['n', 'v']
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
    ]

stemmer = PorterStemmer()
print('词干提取： ', [[stemmer.stem(token) for token in
                       word_tokenize(document)] for document in corpus])

def lemmatize(token, tag):
    if tag[0].lower() in wordnet_tags:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for
                 document in corpus]

print('词形还原： ', [[lemmatize(token, tag) for token, tag in
                       document] for document in tagged_corpus])

# 需要先用 nltk.download() 下载 punkt、averaged_perceptron_tagger
