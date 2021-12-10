import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

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

predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test,
                                                    predictions[:,1])
# 衰退或假阳性率
#                FP
#  Fall-out = ---------
#              TN + FP

# 召回率
#               TP
#  recall = ---------
#            TP + FN

roc_auc = auc(false_positive_rate, recall)   # 参数即为roc_curve返回值前两个

plt.title('受试者操作特征')
plt.plot(false_positive_rate, recall, 'b',
         label='AUC = {:.2f}'.format(roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('召回率')
plt.xlabel('衰退')
plt.show(block=False)
