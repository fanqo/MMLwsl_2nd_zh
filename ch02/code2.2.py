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
# plt.show(block=False)

from sklearn.linear_model import LinearRegression
model = LinearRegression()   # 估计器实例，所有估计器都有fit和predict方法
model.fit(X, y)

# 预测价格
test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]

print('一 %d" pizza的价格应为：$%.2f' % (test_pizza[0,0], predicted_price))

x = np.arange(0, 30, 5).reshape(-1,1)   # 这里reshape不能省，需要一2D数组
plt.plot(x, model.predict(x), color='blue')
plt.show(block=False)
