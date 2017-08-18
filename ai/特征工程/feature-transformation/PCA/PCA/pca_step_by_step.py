# coding:utf-8
"""
1.去掉数据的类别特征（label），将去掉后的d维数据作为样本

2.计算d维的均值向量（即所有数据的每一维向量的均值）

3.计算所有数据的散布矩阵（或者协方差矩阵）

4.计算特征值（e1,e2,...,ed）以及相应的特征向量（lambda1,lambda2,...,lambda d）

5.按照特征值的大小对特征向量降序排序，选择前k个最大的特征向量，组成d*k维的矩阵W（其中每一列代表一个特征向量）

6.运用d*K的特征向量矩阵W将样本数据变换成新的子空间。（用数学式子表达就是，其中x是d*1维的向量，代表一个样本，y是K*1维的在新的子空间里的向量）

动态展示了PCA的过程：http://setosa.io/ev/principal-component-analysis/  写的也很清楚
再推荐一个维基百科的，讲的真的是详细啊https://en.wikipedia.org/wiki/Principal_component_analysis

"""


import numpy as np

"""
1.数据准备----生成三维样本向量

　　首先随机生成40*3维的数据，符合多元高斯分布。假设数据被分为两类，其中一半类别为w1，另一半类别为w2
"""
def getData():
    np.random.seed(4294967295)
    mu_vec1 = np.array([0, 0, 0])
    cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T #依据指定的均值和协方差生成数据
    assert class1_sample.shape == (3, 20)  # 检验数据的维度是否为3*20，若不为3*20，则抛出异常

    mu_vec2 = np.array([1, 1, 1])
    cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T #依据指定的均值和协方差生成数据
    assert class1_sample.shape == (3, 20)  # 检验数据的维度是否为3*20，若不为3*20，则抛出异常
    assert class2_sample.shape == (3, 20)  # 检验数据的维度是否为3*20，若不为3*20，则抛出异常
    return class1_sample,class2_sample


"""
2.作图查看原始数据的分布
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
def rawDataDist(class1_sample,class2_sample):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(class1_sample[0, :], class1_sample[1, :], class1_sample[2, :], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(class2_sample[0, :], class2_sample[1, :], class2_sample[2, :], '^', markersize=8, alpha=0.5, color='red',label='class2')

    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')
    plt.show()


"""
3.去掉数据的类别特征,w1 和w2合并特征
"""
def removeLabel(class1_sample,class2_sample):
    all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
    assert all_samples.shape == (3, 40)  # 检验数据的维度是否为3*20，若不为3*20，则抛出异常
    return  all_samples


"""
4.计算d维向量均值
"""
def meanVector(all_samples):
    mean_x = np.mean(all_samples[0, :])
    mean_y = np.mean(all_samples[1, :])
    mean_z = np.mean(all_samples[2, :])
    mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
    return mean_vector


"""
5.计算散步矩阵或者协方差矩阵
5.1计算散步矩阵
"""

def scatterMatrix(all_samples,mean_vector):
    scatter_matrix = np.zeros((3, 3))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:, i].reshape(3, 1) - mean_vector).dot(
            (all_samples[:, i].reshape(3, 1) - mean_vector).T)

    return scatter_matrix


"""
5.2计算协方差矩阵
"""
def covMatrix ( all_samples):
    cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
    return cov_mat


"""
6.计算相应的特征向量和特征值
6.1.通过计算特征值和特征向量
"""
def scEig(scatter_matrix):
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    return eig_val_sc,eig_vec_sc

"""
6.2.通过协方差矩阵计算特征值和特征向量
"""
def covEig(cov_mat):
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    return eig_val_cov,eig_vec_cov

"""
6.3.验证通过散步矩阵和协方差矩阵得到的特征值和特征向量是否一致
通过散布矩阵和协方差矩阵计算的特征空间相同，协方差矩阵的特征值*39 = 散布矩阵的特征值
"""
def assertEigVec(eig_vec_sc,eig_vec_cov,eig_val_sc,eig_val_cov):
    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:, i].reshape(1, 3).T
        eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T
        assert eigvec_sc.all() == eigvec_cov.all()
        print('Eigenvector {}: \n{}'.format(i + 1, eigvec_sc))
        print('Eigenvalue {} from scatter matrix: {}'.format(i + 1, eig_val_sc[i]))
        print('Eigenvalue {} from covariance matrix: {}'.format(i + 1, eig_val_cov[i]))
        print('Scaling factor: ', eig_val_sc[i] / eig_val_cov[i])
        print(40 * '-')

"""
6.4.快速验证一下特征值-特征向量的计算是否正确
　　得出结果未返回异常，证明计算正确
"""
def validateEig(eig_vec_sc,eig_val_sc,scatter_matrix):
    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[:, i].reshape(1, 3).T
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv, decimal=6, err_msg='', verbose=True)

"""
6.5.可视化特征向量
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def showEig(all_samples,mean_x,mean_y,mean_z,eig_vec_sc):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(all_samples[0, :], all_samples[1, :], all_samples[2, :], 'o', markersize=8, color='green', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    for v in eig_vec_sc.T:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')
    plt.title('Eigenvectors')
    plt.show()

'''
7.根据特征值对特征向量降序排列
我们的目标是减少特征空间的维度，即通过PCA方法将特征空间投影到一个小一点的子空间里，其中特征向量将会构成新的特征空间的轴。
然而，特征向量只会决定轴的方向，他们的单位长度都为1，可以用代码检验一下：
'''
def sortEigVec(eig_vec_sc,eig_val_sc):
    for ev in eig_vec_sc:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        
    # 生成（特征向量，特征值）元祖
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
    # 对（特征向量，特征值）元祖按照降序排列
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # 输出值
    for i in eig_pairs:
        print(i[0])
        
    return eig_pairs


'''
8.选出前k个特征值最大的特征向量
本文的例子是想把三维的空间降维成二维空间，现在我们把前两个最大特征值的特征向量组合起来，生成d*k维的特征向量矩阵W
'''
def bestKEigVec(eig_pairs):
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
    print('Matrix W:\n', matrix_w)
    return matrix_w
    
'''
# 9.将样本转化为新的特征空间
'''
def tranMatrix(all_samples,eig_pairs,):
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
    print('Matrix W:\n', matrix_w)

    transformed = matrix_w.T.dot(all_samples)
    assert transformed.shape == (2, 40), "The matrix is not 2x40 dimensional."

    plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()
    




