import math

math.fabs(-1) ##�����ֵ




##############################################################################
import numpy as np

np.linspace(0, 30, num=50)##�Ȳ�����
np.logspace(0, 30, num=50)##�ȱ�����
np.random.rand([int])##����һ������Ϊ[int]��Ԫ��ֵΪ0-1�������������
np.frompyfunc(func, 1, 1)##��Python��ĺ�������������д�ģ�ת����ufunc���÷���frompyfunc(func, nin, nout)������func����Ҫת���ĺ�����nin�Ǻ�������������ĸ�����nout�Ǵ˺����ķ���ֵ�ĸ�����ע��frompyfunc�����޷���֤���ص��������Ͷ���ȫһ��,��˷���һ���м�����object����Ҫ�ٴ�obj.astype(np.float64)֮�ཫ��Ԫ������ǿ�Ƶ���


import matplotlib as mpl
import matplotlib.pyplot as plt
