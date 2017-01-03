import numpy as np
import pandas as pd
import os

def dataLoaderMat(path):
    files = os.listdir(path)
    data = []
    for f in files:
        df = pd.read_csv(path + '\\' + f, sep = ' ', usecols = [4, 5])
        data.append(df.values[:,0]/df.values[:,1])
    data = np.array([[x for x in y] for y in data])
    return data, len(data)

def dataLoaderVec(path):
    files = os.listdir(path)
    data = np.zeros((0,2))
    for f in files:
        df = pd.read_csv(path + '\\' + f, sep = ' ', usecols = [4, 5])
        data = np.vstack((data, df.values))
    return data[:,0]/data[:,1], data.shape[0]

def slicer(data, sliceLen, overlap):
    N = data.shape[0]
    if (not overlap):
        samples = np.zeros((N//sliceLen,sliceLen))
        for i in range(0,N//sliceLen):
            samples[i,:] = data[i*sliceLen:(1+i)*sliceLen]
        return samples
    samples = np.zeros((N-sliceLen,sliceLen))
    for i in range(0,N-sliceLen):
        samples[i,:] = data[i:i+sliceLen]
    return samples