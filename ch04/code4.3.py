# 词袋模型  <- 包含类似单词的文档经常有相似的含义

# 语料库，包含两个文档
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
    ]

word_list = [line.split() for line in corpus]
import numpy as np
print(np.unique(np.array(word_list)))
