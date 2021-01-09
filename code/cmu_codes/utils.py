import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from classes import FischerfacePredictor,EigenfacePredictor,EigenfacePredictorIllum

def readpgm(name):

    with open(name) as f:
        lines = f.readlines()
    # here,it makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    return (np.array(data[3:]),(data[1],data[0]),data[2])


def loadDataset(img_folder):
    
    train_data = np.zeros((1, 15360))
    train_class = []
    test_data = np.zeros((1, 15360))
    test_class = []
    c = 0 
    np.random.seed(0)

    for dir1 in os.listdir(img_folder):
        choice = np.random.permutation(list(range(0, 32)))
        i=0
        if not os.path.isdir(os.path.join(img_folder, dir1)):
            continue
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1,  file)
            image, img_size, img_max = readpgm(image_path)
            image = np.resize(image,(1, 15360))
            image = image.astype('float32')
            if choice[i]%3==0:
                test_data = np.concatenate((test_data, image), axis = 0)
                test_class.append(c)
            else:
                train_data = np.concatenate((train_data, image), axis = 0)
                train_class.append(c)
            i=i+1
        c=c+1
        
    return train_data[1:] , np.array(train_class), test_data[1:] , np.array(test_class)
