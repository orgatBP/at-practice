# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# GaussianMixtureModel(GMM)����ʵ�ϣ�GMM��k - means���񣬲���GMM��ѧϰ��һЩ�����ܶȺ�����������GMM��������clustering��֮�⣬
# ������������densityestimation�����򵥵�˵��k - means�Ľ����ÿ�����ݵ㱻assign������ĳһ��cluster�ˣ�
# ��GMM�������Щ���ݵ㱻assign��ÿ��cluster�ĸ��ʣ��ֳ���softassignment��
# ��ν��ϸ�˹ģ�ͣ�GMM������ָ�������ĸ����ܶȷֲ����й��ƣ������Ʋ��õ�ģ�ͣ�ѵ��ģ�ͣ��Ǽ�����˹ģ�͵ļ�Ȩ�ͣ������Ǽ���Ҫ��ģ��ѵ��ǰ�����ã���
# ÿ����˹ģ�;ʹ�����һ���ࣨһ��Cluster�����������е����ݷֱ��ڼ�����˹ģ����ͶӰ���ͻ�ֱ�õ��ڸ������ϵĸ��ʡ�Ȼ�����ǿ���ѡȡ������������Ϊ�о������

#GaussianMixture(n_components=1, covariance_type=��full��, tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params=��kmeans��, weights_init=None, means_init=None, 
#precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10) 
#������ 
#1. n_components:��ϸ�˹ģ�͸�����Ĭ��Ϊ1 
#2. covariance_type:Э�������ͣ�����{��full��,��tied��, ��diag��, ��spherical��}���֣��ֱ��Ӧ��ȫЭ�������Ԫ�ض���Ϊ�㣩����ͬ����ȫЭ�������HMM���õ������Խ�Э������󣨷ǶԽ�Ϊ�㣬�Խǲ�Ϊ�㣩������Э������󣨷ǶԽ�Ϊ�㣬�Խ���ȫ��ͬ���������ԣ���Ĭ�ϡ�full�� ��ȫЭ������� 
#3. tol��EM����ֹͣ��ֵ��Ĭ��Ϊ1e-3. 
#4. reg_covar:Э����ԽǷǸ����򻯣���֤Э��������Ϊ����Ĭ��Ϊ0 
#5. max_iter:������������Ĭ��100 
#6. n_init:��ʼ�����������ڲ�����ѳ�ʼ������Ĭ��Ϊ1 
#7. init_params: {��kmeans��, ��random��}, defaults to ��kmeans��.��ʼ������ʵ�ַ�ʽ��Ĭ����kmeansʵ�֣�Ҳ����ѡ��������� 
#8. weights_init:�����ģ�͵�����Ȩ�أ������Լ��裬Ĭ�ϰ���7���� 
#9. means_init:��ʼ����ֵ��ͬ8 
#10. precisions_init:��ʼ����ȷ�ȣ�ģ�͸�����������������Ĭ�ϰ���7ʵ�� 
#11. random_state :����������� 
#12. warm_start :��ΪTrue����fit�������û�����һ��fit�����Ľ����Ϊ��ʼ���������ʺ���ͬ������fit��������ܼ���������Ĭ��ΪFalse�� 
#13. verbose :ʹ�ܵ�����Ϣ��ʾ��Ĭ��Ϊ0������Ϊ1���ߴ���1����ʾ����Ϣ��ͬ�� 
#14. verbose_interval :��13�ҹ�����ʹ�ܵ�����Ϣ��ʾ�����ö��ٴε�������ʾ��Ϣ��Ĭ��10�Ρ�


if __name__ == '__main__':
    style = 'sklearn'

    np.random.seed(0)
    mu1_fact = (0, 0, 0)
    # ���ɶԽǾ���
    cov1_fact = np.diag((1, 2, 3))
    data1 = np.random.multivariate_normal(mu1_fact, cov1_fact, 400)
    mu2_fact = (2, 2, 1)
    cov2_fact = np.array(((1, 1, 3), (1, 2, 1), (0, 0, 1)))
    data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 100)
    data = np.vstack((data1, data2))
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':

        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        g.fit(data)
        print u'������:\t', g.weights_[0]
        print u'��ֵ:\n', g.means_, '\n'
        print u'����:\n', g.covariances_, '\n'
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        n, d = data.shape
        # ���ָ��
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print i, ":\t", mu1, mu2
        print u'������:\t', pi
        print u'��ֵ:\t', mu1, mu2
        print u'����:\n', sigma1, '\n\n', sigma2, '\n'

    # Ԥ�����
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(10, 5), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', edgecolors='k', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'ԭʼ����', fontsize=15)
    ax = fig.add_subplot(122, projection='3d')
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    print order
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    print u'׼ȷ�ʣ�%.2f%%' % (100 * acc)
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', edgecolors='k', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', edgecolors='k', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM�㷨����', fontsize=15)
    plt.suptitle(u'EM�㷨��ʵ��', fontsize=18)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.show()
