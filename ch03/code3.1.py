import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']

# Note： 此处数据与表中有出入
X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
    ])

y_train = ['male', 'male', 'male', 'male', 'female', 'female',
           'female', 'female', 'female']

plt.figure()
plt.title('不同性别人类身高和体重')
plt.xlabel('身高 (cm)')
plt.ylabel('体重 (kg)')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
    # 使用不同的符号来标识男女

plt.grid(True)
plt.show(block=False)
