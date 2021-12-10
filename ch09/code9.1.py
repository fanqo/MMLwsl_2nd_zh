import numpy as np

# 样本，10个随机整数
sample = np.random.randint(low = 1, high = 100, size = 10)
print('初始样本： {}'.format(sample))
print('样本均值： {}'.format(sample.mean()))

# 自助法(bootstrap)重采样100次，从sample的数据中再采样
resamples = [np.random.choice(sample, size=sample.shape)
             for i in range(100)]
print('自助采样次数： {}'.format(len(resamples)))
print('重新采样示例： {}'.format(resamples[0]))

resample_means = np.array([resample.mean() for resample in resamples])
print('重新采样均值的均值： {}'.format(resample_means.mean()))
