import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('SMSSpamCollection', delimiter='\t',
                 names=['label', 'message'])

# 把label替换为0和1才可运行precision等评分
df['label'].replace('spam', 1, inplace=True)
df['label'].replace('ham', 0, inplace=True)

X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(df['message'], df['label'], random_state = 11)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('准确率：\n\t {}'.format(scores))
print('平均准确率： {}'.format(scores.mean()))

# 若类别的比例在总样本中呈偏态分布，准确率并不是很有效
#                     TP + TN
#   accuracy =  -------------------
#                TP + TP + FP + FN

precisions = cross_val_score(classifier, X_train, y_train, cv=5,
                             scoring='precision')
print('精确率：\n\t {}'.format(precisions))
print('平均精确率： {}'.format(precisions.mean()))

# 精确率
#                  TP
#  precision = ----------
#               TP + FP

recalls = cross_val_score(classifier, X_train, y_train, cv=5,
                          scoring='recall')
print('召回率： \n\t {}'.format(recalls))
print('平均召回率： {}'.format(recalls.mean()))

# 召回率
#               TP
#  recall = ---------
#            TP + FN

f1s = cross_val_score(classifier, X_train, y_train, cv=5,
                      scoring='f1')
print('平均F1得分为： {}'.format(f1s.mean()))
