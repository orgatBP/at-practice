# coding:utf-8
"""
1.ȥ�����ݵ����������label������ȥ�����dά������Ϊ����

2.����dά�ľ�ֵ���������������ݵ�ÿһά�����ľ�ֵ��

3.�����������ݵ�ɢ�����󣨻���Э�������

4.��������ֵ��e1,e2,...,ed���Լ���Ӧ������������lambda1,lambda2,...,lambda d��

5.��������ֵ�Ĵ�С������������������ѡ��ǰk�������������������d*kά�ľ���W������ÿһ�д���һ������������

6.����d*K��������������W���������ݱ任���µ��ӿռ䡣������ѧʽ�ӱ����ǣ�����x��d*1ά������������һ��������y��K*1ά�����µ��ӿռ����������

��̬չʾ��PCA�Ĺ��̣�http://setosa.io/ev/principal-component-analysis/  д��Ҳ�����
���Ƽ�һ��ά���ٿƵģ������������ϸ��https://en.wikipedia.org/wiki/Principal_component_analysis

"""


import numpy as np

"""
1.����׼��----������ά��������

���������������40*3ά�����ݣ����϶�Ԫ��˹�ֲ����������ݱ���Ϊ���࣬����һ�����Ϊw1����һ�����Ϊw2
"""
def getData():
    np.random.seed(4294967295)
    mu_vec1 = np.array([0, 0, 0])
    cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T #����ָ���ľ�ֵ��Э������������
    assert class1_sample.shape == (3, 20)  # �������ݵ�ά���Ƿ�Ϊ3*20������Ϊ3*20�����׳��쳣

    mu_vec2 = np.array([1, 1, 1])
    cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T #����ָ���ľ�ֵ��Э������������
    assert class1_sample.shape == (3, 20)  # �������ݵ�ά���Ƿ�Ϊ3*20������Ϊ3*20�����׳��쳣
    assert class2_sample.shape == (3, 20)  # �������ݵ�ά���Ƿ�Ϊ3*20������Ϊ3*20�����׳��쳣
    return class1_sample,class2_sample


"""
2.��ͼ�鿴ԭʼ���ݵķֲ�
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
3.ȥ�����ݵ��������,w1 ��w2�ϲ�����
"""
def removeLabel(class1_sample,class2_sample):
    all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
    assert all_samples.shape == (3, 40)  # �������ݵ�ά���Ƿ�Ϊ3*20������Ϊ3*20�����׳��쳣
    return  all_samples


"""
4.����dά������ֵ
"""
def meanVector(all_samples):
    mean_x = np.mean(all_samples[0, :])
    mean_y = np.mean(all_samples[1, :])
    mean_z = np.mean(all_samples[2, :])
    mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
    return mean_vector


"""
5.����ɢ���������Э�������
5.1����ɢ������
"""

def scatterMatrix(all_samples,mean_vector):
    scatter_matrix = np.zeros((3, 3))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:, i].reshape(3, 1) - mean_vector).dot(
            (all_samples[:, i].reshape(3, 1) - mean_vector).T)

    return scatter_matrix


"""
5.2����Э�������
"""
def covMatrix ( all_samples):
    cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
    return cov_mat


"""
6.������Ӧ����������������ֵ
6.1.ͨ����������ֵ����������
"""
def scEig(scatter_matrix):
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    return eig_val_sc,eig_vec_sc

"""
6.2.ͨ��Э��������������ֵ����������
"""
def covEig(cov_mat):
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    return eig_val_cov,eig_vec_cov

"""
6.3.��֤ͨ��ɢ�������Э�������õ�������ֵ�����������Ƿ�һ��
ͨ��ɢ�������Э����������������ռ���ͬ��Э������������ֵ*39 = ɢ�����������ֵ
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
6.4.������֤һ������ֵ-���������ļ����Ƿ���ȷ
�����ó����δ�����쳣��֤��������ȷ
"""
def validateEig(eig_vec_sc,eig_val_sc,scatter_matrix):
    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[:, i].reshape(1, 3).T
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv, decimal=6, err_msg='', verbose=True)

"""
6.5.���ӻ���������
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
7.��������ֵ������������������
���ǵ�Ŀ���Ǽ��������ռ��ά�ȣ���ͨ��PCA�����������ռ�ͶӰ��һ��Сһ����ӿռ�����������������ṹ���µ������ռ���ᡣ
Ȼ������������ֻ�������ķ������ǵĵ�λ���ȶ�Ϊ1�������ô������һ�£�
'''
def sortEigVec(eig_vec_sc,eig_val_sc):
    for ev in eig_vec_sc:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        
    # ���ɣ���������������ֵ��Ԫ��
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
    # �ԣ���������������ֵ��Ԫ�水�ս�������
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # ���ֵ
    for i in eig_pairs:
        print(i[0])
        
    return eig_pairs


'''
8.ѡ��ǰk������ֵ������������
���ĵ������������ά�Ŀռ併ά�ɶ�ά�ռ䣬�������ǰ�ǰ�����������ֵ�����������������������d*kά��������������W
'''
def bestKEigVec(eig_pairs):
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
    print('Matrix W:\n', matrix_w)
    return matrix_w
    
'''
# 9.������ת��Ϊ�µ������ռ�
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
    




