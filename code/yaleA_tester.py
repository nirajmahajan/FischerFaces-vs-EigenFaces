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

from utils import getAllErrors, getAllErrorsLeavingOne
from dataloaders import YaleDataset

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = "../datasets/yalefaces/")
parser.add_argument('--seed', type = int, default = 0)
args = parser.parse_args()

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

dataset = YaleDataset(args.path)

fischer_error, eigen_error, eigen_illu_error = getAllErrors(dataset.Xtrain, dataset.Ytrain, dataset.Xtest, dataset.Ytest)
print("Error Rates:")
print("Fischer Predictor                            --> {}".format(100*(fischer_error)))
print("Eigen Predictor                              --> {}".format(100*(eigen_error)))
print("Eigen Predictor (without top 3 eigvectors)   --> {}".format(100*(eigen_illu_error)))


print("\nGlasses Predictor (Using leaving one out method):")
# fischer_error, eigen_error, eigen_illu_error = getAllErrors(dataset.X_glasses[0:18,:], dataset.Y_glasses[0:18], dataset.X_glasses[18:,:], dataset.Y_glasses[18:], parameters = [1,10,10])
fischer_error, eigen_error, eigen_illu_error = getAllErrorsLeavingOne(dataset.X_glasses, dataset.Y_glasses)
print("Fischer Predictor                            --> {}".format(100*(fischer_error)))
print("Eigen Predictor                              --> {}".format(100*(eigen_error)))
print("Eigen Predictor (without top 3 eigvectors)   --> {}".format(100*(eigen_illu_error)))