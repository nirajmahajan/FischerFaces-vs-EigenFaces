import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

class PCA(object):
	"""takes a nxd input matrix X and stores a PCA model to transform to a k dimensional space"""
	def __init__(self, n_components = 1):
		super(PCA, self).__init__()
		self.n_components = n_components

	def fit(self, X, illum = False):
		self.mean = X.mean(0)
		self.insize = X.shape[1]
		n_data = X.shape[0]
		assert(self.n_components <= n_data)
		X_norm = X - self.mean.reshape(1,-1)
		mat = (X_norm@X_norm.T)/(n_data - 1)
		[eigvals, eigvecs] = np.linalg.eigh(mat)
		indices = np.argsort(eigvals)[::-1]
		self.eigvals_all = eigvals[indices]
		if illum:
			indices = indices[3:self.n_components+3]
		else:
			indices = indices[:self.n_components]
		eigvals = eigvals[indices]
		eigvecs_big = X.T@eigvecs
		norms = (eigvecs_big**2).sum(0).reshape(1,-1) ** 0.5
		eigvecs_big = eigvecs_big/(norms+1e-8)
		# dxn_components
		self.eigvals_chosen = eigvals
		self.transform_mat = eigvecs_big[:,indices]

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)

	def transform(self, X):
		# mxd 
		assert(self.insize == X.shape[1])
		return -(X - self.mean.reshape(1,-1))@self.transform_mat


class FLD(object):
	"""An object for computing the fischer linear discriminant of the input array"""
	def __init__(self, out_dim = 1):
		super(FLD, self).__init__()
		self.out_dim = out_dim

	# X 	 -> nxd
	# labels -> n,
	# Assuming labels of the form {0,1,2,3...c-1}
	def fit(self, X, labels):
		self.in_dim = X.shape[1]
		self.n_data = X.shape[0]
		self.c = max(labels)+1
		# error check
		for i in range(self.c):
			if not i in labels:
				print("Class {} not present in labels".format(i), flush = True)
				os._exit(-1)

		# generate classwise X and number of elements per class (Ni)
		classwise_X = []
		MUi = []
		Ni = []
		for i in range(self.c):
			indices = (labels == 1)
			indices = 1*indices  #converting to bool to int
			classwise_X.append(X[indices,:])
			Ni.append(np.sum(indices))
			MUi.append(classwise_X[i].mean(0).reshape(-1,1))

		# generate Sb, Sw
		self.mu_global = X.mean(0).reshape(-1,1)
		self.Sb = np.zeros((self.in_dim,self.in_dim))
		self.Sw = np.zeros((self.in_dim,self.in_dim))
		for i in range(self.c):
			vec = MUi[i] - self.mu_global
			Xi_norm = classwise_X[i] - MUi[i].reshape(1,-1)
			self.Sb += Ni[i]*(vec@vec.T)
			self.Sw += (Xi_norm.T@Xi_norm)

		# get the generalised eigvals
		# self.eigvals_gen, self.W = scipy.linalg.eigh(self.Sb,self.Sw,eigvals_only=False,subset_by_value=[-np.inf, 10])
		self.eigvals_gen, self.W = scipy.linalg.eig(self.Sb,self.Sw)

		indices = np.argsort(self.eigvals_gen)[::-1]
		self.eigvals_gen_all = self.eigvals_gen[indices]
		indices = indices[:self.out_dim]
		self.eigvals_gen = self.eigvals_gen[indices]
		self.W = self.W[:,indices]

	def transform(self, X):
		return X@self.W


class Fischerfaces(object):
	"""docstring for Fischerfaces"""
	def __init__(self, out_dim = 1):
		super(Fischerfaces, self).__init__()
		self.out_dim = out_dim

	def fit(self, X, labels):
		self.in_dim = X.shape[1]
		self.n_data = X.shape[0]
		self.c = max(labels)+1
		# error check
		for i in range(self.c):
			if not i in labels:
				print("Class {} not present in labels".format(i), flush = True)
				os._exit(-1)

		self.pca = PCA(n_components = self.n_data-self.c)
		self.fld = FLD(out_dim = self.out_dim)
		
		lowerX = self.pca.fit_transform(X)
		self.fld.fit(lowerX, labels)

	def transform(self, X):
		return self.fld.transform(self.pca.transform(X))


class FischerfacePredictor(object):
	"""docstring for FischerfacePredictor"""
	def __init__(self, out_dim):
		super(FischerfacePredictor, self).__init__()
		self.out_dim = out_dim

	def train(self, X, labels):
		self.fischerfaces_model = Fischerfaces(out_dim = self.out_dim)
		self.fischerfaces_model.fit(X, labels)
		self.train_embeddings = self.fischerfaces_model.transform(X)
		self.train_labels = labels

	def test(self, X):
		# X mxd -> mxk -> 1xmxk
		temp_test = np.expand_dims(self.fischerfaces_model.transform(X), 0)
		# X nxd -> nxk -> nx1xk
		temp_train = np.expand_dims(self.train_embeddings, 1)
		diff = ((temp_train - temp_test)**2).sum(2)
		min_pos = np.argmin(diff, 0)
		return self.train_labels[min_pos]

class EigenfacePredictor(object):
	"""docstring for EigenfacePredictor"""
	def __init__(self, out_dim):
		super(EigenfacePredictor, self).__init__()
		self.out_dim = out_dim

	def train(self, X, labels):
		self.eigenfaces_model = PCA(n_components = self.out_dim)
		self.eigenfaces_model.fit(X)
		self.train_embeddings = self.eigenfaces_model.transform(X)
		self.train_labels = labels

	def test(self, X):
		# X mxd -> mxk -> 1xmxk
		temp_test = np.expand_dims(self.eigenfaces_model.transform(X), 0)
		# X nxd -> nxk -> nx1xk
		temp_train = np.expand_dims(self.train_embeddings, 1)
		diff = ((temp_train - temp_test)**2).sum(2)
		min_pos = np.argmin(diff, 0)
		return self.train_labels[min_pos]

class EigenfacePredictorIllum(object):
	"""docstring for EigenfacePredictorIllum"""
	def __init__(self, out_dim):
		super(EigenfacePredictorIllum, self).__init__()
		self.out_dim = out_dim

	def train(self, X, labels):
		self.eigenfaces_model = PCA(n_components = self.out_dim)
		self.eigenfaces_model.fit(X, illum = True)
		self.train_embeddings = self.eigenfaces_model.transform(X)
		self.train_labels = labels

	def test(self, X):
		# X mxd -> mxk -> 1xmxk
		temp_test = np.expand_dims(self.eigenfaces_model.transform(X), 0)
		# X nxd -> nxk -> nx1xk
		temp_train = np.expand_dims(self.train_embeddings, 1)
		diff = ((temp_train - temp_test)**2).sum(2)
		min_pos = np.argmin(diff, 0)
		return self.train_labels[min_pos]
