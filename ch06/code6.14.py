import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score

def print_true_pred(y_true, y_pred):
    print('真实值为：')
    print('\t'+str(y_true).replace('\n','\n\t'))
    print('预测值为：')
    print('\t'+str(y_pred).replace('\n','\n\t'))

# Hamming 损失：不正确标签的平均比例，完美得分为0
def p_hamming():
    print_true_pred(y_true, y_pred)
    print('Hamming损失为：')
    print('\t', hamming_loss(y_true, y_pred))
    print('-----')
    
y_true = np.array([[0.0, 1.0], [1.0, 1.0]])
y_pred = np.array([[0.0, 1.0], [1.0, 1.0]])
#print_true_pred(y_true, y_pred)
p_hamming()

y_pred = np.array([[1.0, 1.0], [1.0, 1.0]])
p_hamming()

y_pred = np.array([[1.0, 1.0], [0.0, 1.0]])
p_hamming()

# Jaccard 相似系数：预测标签与真实标签的交集 除以 预测标签与真实标签补集 （数目比）
# intersection over union
def p_jaccard():
    print_true_pred(y_true, y_pred)
    print('Jaccard相似系数为：')
    print('\t', jaccard_score(y_true, y_pred, average='micro'))
    print('-----')

y_pred = np.array([[0.0, 1.0], [1.0, 1.0]])
p_jaccard()

y_pred = np.array([[1.0, 1.0], [1.0, 1.0]])
p_jaccard()

y_pred = np.array([[1.0, 1.0], [0.0, 1.0]])
p_jaccard()
