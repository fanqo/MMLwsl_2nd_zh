from sklearn.linear_model import LinearRegression

# 解释变量，这里不用加一列1
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
# 响应变量
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)

X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]

predictions = model.predict(X_test)
# enumerate 返回 (0, seq[0]), (1, seq[1]) ...
for i, prediction in enumerate(predictions):
    print('预测: {}, 目标： {}'.format(prediction, y_test[i]))
    print('R^2: {:.2f}'.format(model.score(X_test, y_test)))



