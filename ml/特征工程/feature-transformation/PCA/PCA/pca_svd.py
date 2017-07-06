'''
@author: Alex Tian
'''
from numpy import *
import matplotlib.pyplot as plt


import numpy as np
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])

mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])

import matplotlib.pyplot as plt
plt.plot(scaled_x,scaled_y,'o')

cov=np.cov(scaled_x,scaled_y)

eig_val, eig_vec = np.linalg.eig(cov)

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature=eig_pairs[0][1]

new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))

print eig_pairs
u,m,v=linalg.svd(cov) #用SVD 快速求解
print u,m,v
print new_data_reduced



