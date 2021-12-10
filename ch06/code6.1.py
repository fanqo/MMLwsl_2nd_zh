# 垃圾短信分类
import pandas as pd
# 数据可从如下链接下载
# https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
#       0                                                  1
# 0   ham  Go until jurong point, crazy.. Available only ...
# 1   ham                      Ok lar... Joking wif u oni...
print('spam 短信的数目： {}'.format(df[df[0] == 'spam'][0].count()))
print('ham 短信的数目： {}'.format(df[df[0] == 'ham'][0].count()))
