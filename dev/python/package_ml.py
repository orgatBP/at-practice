import math

math.fabs(-1) ##求绝对值




##############################################################################
import numpy as np

np.linspace(0, 30, num=50)##等差数列
np.logspace(0, 30, num=50)##等比数列
np.random.rand([int])##产生一个长度为[int]，元素值为0-1的随机数的数组
np.frompyfunc(func, 1, 1)##把Python里的函数（可以是自写的）转化成ufunc，用法是frompyfunc(func, nin, nout)，其中func是需要转换的函数，nin是函数的输入参数的个数，nout是此函数的返回值的个数。注意frompyfunc函数无法保证返回的数据类型都完全一致,因此返回一个中间类型object，需要再次obj.astype(np.float64)之类将其元素类型强制调齐


import matplotlib as mpl
import matplotlib.pyplot as plt
