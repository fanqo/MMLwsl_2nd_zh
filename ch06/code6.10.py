# import zipfile   # read in zip file directly
import pandas as pd

# 数据可从
# https://github.com/benbobyabraham/kaggle-sentiment_analysis_movie_review_NLP
# 下载
df = pd.read_csv('train.tsv.zip', compression='zip', delimiter='\t', header=0)

# 所有non-NA/null的观察的数目
print(df.count())

# 前面几行
print(df.head())

# Sentiment列包含响应变量，0对应情绪负向, 1对应略负向,..., 4正向

# 显示一些短语
print(df['Phrase'].head(10))

