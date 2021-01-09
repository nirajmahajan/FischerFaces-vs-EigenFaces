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
from utils import *

tr, tr_c, ts, ts_c=loadDataset('../../datasets/cmu/')

model_fischer = FischerfacePredictor(19)
model_fischer.train(tr, tr_c)
prediction_fischer = model_fischer.test(ts)

model_eigen = EigenfacePredictor(19)
model_eigen.train(tr, tr_c)
prediction_eigen = model_eigen.test(ts)

model_eigen_illum = EigenfacePredictorIllum(19)
model_eigen_illum.train(tr, tr_c)
prediction_eigen_illum = model_eigen_illum.test(ts)

print("Error Rates:")
print("Fischer Predictor                            --> {}".format(100*(1 - np.sum(1*(prediction_fischer==ts_c))/ts_c.size)))
print("Eigen Predictor                              --> {}".format(100*(1 - np.sum(1*(prediction_eigen==ts_c))/ts_c.size)))
print("Eigen Predictor (without top 3 eigvectors)   --> {}".format(100*(1 - np.sum(1*(prediction_eigen_illum==ts_c))/ts_c.size)))


