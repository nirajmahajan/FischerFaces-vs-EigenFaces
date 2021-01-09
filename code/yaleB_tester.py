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
from dataloaders import YaleBDataset

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = "../datasets/yaleB/")
parser.add_argument('--seed', type = int, default = 0)
args = parser.parse_args()

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

dataset = YaleBDataset(args.path)

fischer_error, eigen_error, eigen_illu_error = getAllErrors(dataset.Xtrain, dataset.Ytrain, dataset.Xtest, dataset.Ytest, parameters = [38,50,50])
print("Error Rates:")
print("Fischer Predictor                            --> {}".format(100*(fischer_error)))
print("Eigen Predictor                              --> {}".format(100*(eigen_error)))
print("Eigen Predictor (without top 3 eigvectors)   --> {}".format(100*(eigen_illu_error)))
