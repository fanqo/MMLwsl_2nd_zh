# 导入numpy、matplotlib.pyploty，并分别赋予它们别名np、plt以方便后面使用
import numpy as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']

# 用大写字母表示矩阵，小写字母表示矢量，X-pizza直径，y-pizza价格
X = np.array([[6], [8], [10], [14], [18]]).reshape(-1,1)   # 这里 reshape 可无
y = [7, 9, 13, 17.5, 18]

plt.figure()
plt.title('pizza价格随直径的变化')
plt.xlabel('直径 (英寸)')
plt.ylabel('价格 (美元)')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

