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

seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


###### following is the usage
# a = FischerfacePredictor(c-1)
# data = DATA
# labels = np.arange(500)//50
# a.train(data, labels)

# prediction = a.test(data)

###### following is the usage
a = EigenfacePredictor(9)
data = np.random.randn(500,15)
labels = np.arange(500)//50
a.train(data, labels)

prediction = a.test(data)