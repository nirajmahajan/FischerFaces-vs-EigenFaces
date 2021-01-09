import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA as PP
import matplotlib.pyplot as plt
import os
import argparse
import sys
import time
import random

from classes import FischerfacePredictor,EigenfacePredictor,EigenfacePredictorIllum

# first parameter should be the number of classes
def getAllErrors(Xtrain, Ytrain, Xtest, Ytest, parameters = [15,30,30]):
	fischer_model = FischerfacePredictor(parameters[0])
	fischer_model.train(Xtrain, Ytrain)

	prediction = fischer_model.test(Xtest)
	fischer_accuracy = (prediction == Ytest).sum()/Ytest.shape[0]

	eigen_model = EigenfacePredictor(parameters[1])
	eigen_model.train(Xtrain, Ytrain)

	prediction = eigen_model.test(Xtest)
	eigen_accuracy = (prediction == Ytest).sum()/Ytest.shape[0]

	eigen_illu_model = EigenfacePredictorIllum(parameters[2])
	eigen_illu_model.train(Xtrain, Ytrain)

	prediction = eigen_illu_model.test(Xtest)
	eigen_illu_accuracy = (prediction == Ytest).sum()/Ytest.shape[0]

	return 1-fischer_accuracy, 1-eigen_accuracy, 1-eigen_illu_accuracy

# to be used for yale, cmu, not yaleB
def getAllErrorsLeavingOne(X, Y, parameters = [1,10,10]):
	fischer_error = 0
	eigen_error = 0
	eigen_illu_error = 0
	for i in range(Y.shape[0]//2):
		tempX = np.delete(X,[2*i,(2*i)+1],0)
		tempY = np.delete(Y,[2*i,(2*i)+1],0)
		fe, ee, eie = getAllErrors(tempX, tempY, X[2*i:(2*i)+1,:], Y[2*i:(2*i)+1], parameters = parameters)
		fischer_error += fe*2
		eigen_error += ee*2
		eigen_illu_error += eie*2

	fischer_error /= Y.shape[0]
	eigen_error /= Y.shape[0]
	eigen_illu_error /= Y.shape[0]

	return fischer_error, eigen_error, eigen_illu_error

