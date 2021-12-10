# 混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = \
    [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']

y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
confusion_matrix = confusion_matrix(y_test, y_pred)
print('混淆矩阵：')
print('\t'+str(confusion_matrix).replace('\n', '\n\t'))

plt.matshow(confusion_matrix)
plt.title('混淆矩阵')
plt.colorbar()
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show(block=False)
