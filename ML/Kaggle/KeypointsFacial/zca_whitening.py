import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
# import seaborn as sns
# sns.set(color_codes=True)

# In order to apply ZCA whitening over a dataset
# you can use this loop:
# 
## for i in range(X.shape[0]):
##	 x = X[i]
## 	 x = x.reshape(96, 96)
## 	 x = zca(x, epsilon=ep)
## 	 X[i] = x.reshape(1, 96, 96)


# def zca_whitening(X, epsilon=1e-5):
# 	X -= np.mean(X, axis = 0)
# 	C = np.cov(X, rowvar=True)
# 	U, S, V = np.linalg.svd(C, full_matrices=False)
# 	ZCAwhiten = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S ** 2 + epsilon)), U.T))
# 	return np.dot(ZCAwhiten, X)

# def zca_sk(inputs):
# 	pca = PCA(whiten=True)
# 	transformed = pca.fit_transform(inputs)
# 	pca.whiten = False
# 	return pca.inverse_transform(transformed)

def zca(inputs, epsilon=1e-5):
    #Correlation matrix
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1]

    #Singular Value Decomposition
    U,S,V = np.linalg.svd(sigma)

    #ZCA Whitening matrix
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T)

    #Data whitening
    return np.dot(ZCAMatrix, inputs)

def zca_data(X, epsilon=1e-5):
	for i in range(X.shape[0]):
		x = X[i]
		x = x.reshape(96, 96)
		x = zca(x, epsilon=epsilon)
		X[i] = x.reshape(1, 96, 96)
	return X

########
# TEST #
########
# from LoadData import load
# X, y = load()

# # i = np.random.random_integers(0, high=X.shape[0] - 1)
# ep = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# x2 = [0 for i in range(8)]
# i = 0
# for e in ep:
#         x = np.copy(X[4])
#         x = x.reshape(96, 96)

#         # ep = 1e-5
#         print e
#         # x1 = zca_whitening(x, epsilon=ep)
#         x2[i] = zca(x, epsilon=e)
#         i += 1
#         # x3 = zca_sk(x)

# print len(x2)

# fig = plt.figure(figsize=(20,3))
# fig.suptitle('Epsilon effect (from 1 to 1e-7)', fontsize=15)
# ax1 = plt.subplot2grid((1, 9), (0,0), colspan=1, xticks=[], yticks=[])
# ax2 = plt.subplot2grid((1, 9), (0,1), colspan=1, xticks=[], yticks=[])
# ax3 = plt.subplot2grid((1, 9), (0,2), colspan=1, xticks=[], yticks=[])
# ax4 = plt.subplot2grid((1, 9), (0,3), colspan=1, xticks=[], yticks=[])
# ax5 = plt.subplot2grid((1, 9), (0,4), colspan=1, xticks=[], yticks=[])
# ax6 = plt.subplot2grid((1, 9), (0,5), colspan=1, xticks=[], yticks=[])
# ax7 = plt.subplot2grid((1, 9), (0,6), colspan=1, xticks=[], yticks=[])
# ax8 = plt.subplot2grid((1, 9), (0,7), colspan=1, xticks=[], yticks=[])
# ax9 = plt.subplot2grid((1, 9), (0,8), colspan=1, xticks=[], yticks=[])
# # ax1 = fig.add_subplot(221, xticks=[], yticks=[])
# # ax2 = fig.add_subplot(222, xticks=[], yticks=[])
# # ax3 = fig.add_subplot(221)
# # ax4 = fig.add_subplot(144, xticks=[], yticks=[])
# ax1.imshow(X[4].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax2.imshow(x2[0].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax3.imshow(x2[1].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax4.imshow(x2[2].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax5.imshow(x2[3].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax6.imshow(x2[4].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax7.imshow(x2[5].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax8.imshow(x2[6].reshape(96, 96), cmap = plt.get_cmap('gray'))
# ax9.imshow(x2[7].reshape(96, 96), cmap = plt.get_cmap('gray'))
# # ax3.plot(S)
# # ax4.imshow(x3.reshape(96, 96), cmap = plt.get_cmap('gray'))
# # plt.legend()

# # plt.show()
# plt.savefig('ZCAwhitening_epsilon.png')
