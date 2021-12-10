# 垃圾短信分类
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 数据可从如下链接下载
# https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
#       0                                                  1
# 0   ham  Go until jurong point, crazy.. Available only ...
# 1   ham                      Ok lar... Joking wif u oni...

X = df[1].values
y = df[0].values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
for i, prediction in enumerate(predictions[:5]):
    print('预测结果： {}, 短信内容： {}'.format(prediction, X_test_raw[i]))

    

