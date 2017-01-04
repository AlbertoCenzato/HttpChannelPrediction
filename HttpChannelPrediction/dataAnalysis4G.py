import numpy as np

from utils import dataLoaderMat, dataLoaderVec, slicer
from crossValidation import chooseBestSplitLen

 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

data, N = dataLoaderVec('tram')

from sklearn import linear_model as lm

linReg = lm.LinearRegression()
len = chooseBestSplitLen(data, linReg, max_size = 30, plot = True)

'''
# Here we predict a completely new trajectory out of the old ones

data, N = dataLoaderMat('tram')

train_size = 0.66
Ntr = int(np.ceil(N*train_size))
train = np.array([])
validation = np.array([])
for i in range(0,Ntr):
    train = np.append(train,data[i])
for i in range(Ntr,N):
    validation = np.append(validation,data[i])

sliceLen = 50
samples_tr = slicer(train,sliceLen, True)
samples_va = slicer(validation, sliceLen, True)

Xtr,Ytr = samples_tr[:, :sliceLen-2],samples_tr[:,sliceLen-1]
Xva,Yva = samples_va[:, :sliceLen-2],samples_va[:,sliceLen-1]

from sklearn import linear_model as lm
linReg = lm.LinearRegression(normalize=True)
linReg.fit(Xtr, Ytr)
print 'Training error: ', 1-linReg.score(Xtr,Ytr)
print 'Validation error: ', 1-linReg.score(Xva,Yva)

import matplotlib.pyplot as pl
pl.figure()
pl.plot(Yva[:100], 'b')
pl.plot(linReg.predict(Xva)[:100],'r')
pl.show()

pl.figure()
pl.plot(validation[:2000], 'b')
pl.show()
'''

# same as above, using cars

data, N = dataLoaderVec('car')

linReg = lm.LinearRegression()
len = chooseBestSplitLen(data, linReg, max_size = 30, plot = True)

#-------------------------------------------
#------------ USING SVMs -------------------
#-------------------------------------------

#----- carica i dati ----- 
data, N = dataLoaderVec('tram')
'''
train_size = 0.66
Ntr = int(np.ceil(N*train_size))
X = np.array([])
train = np.array([])
validation = np.array([])
for i in range(0,N):
    X = np.append(X,data[i])
    
#----- normalizzazione ------
from sklearn import preprocessing
X = preprocessing.scale(X)

for i in range(0,N):
    if(i < Ntr):
        train = np.append(train,data[i])
    else:
        validation = np.append(validation,data[i])

#print train.shape

#----- slicing -----
sliceLen = 30
samples_tr = slicer(train,sliceLen, False)
samples_va = slicer(validation, sliceLen, False)

Xtr,Ytr = samples_tr[:, :sliceLen-2],samples_tr[:,sliceLen-1]
Xva,Yva = samples_va[:, :sliceLen-2],samples_va[:,sliceLen-1]
'''

#----- training -----
testPerc = 0.66

from sklearn import svm
from sklearn.cross_validation import train_test_split

minCerr = []
minS = []
for c in [0.01,0.1,1,10,100,1000,10000]:
    print('-----------------ITERAZIONE CON C = ', c, '----------------')
    svr = svm.LinearSVR(verbose = True, C = c)
    sliceLenOpt = chooseBestSplitLen(data, svr, max_size = 10)
    #if sliceLenOpt == 1 :           #questa riga e la prossima le avevo aggiunte io, perchè ho notato che a volte il sliceLenOpt è uguale a 1 
     #   sliceLenOpt = sliceLenOpt + 1          #e la cosa dà problemi perchè le matrici Xtr Xva etc vengono fuori con 0 colonne- non ha funzionato però
    np.append(minS, min)
    samples = slicer(data, sliceLenOpt, True)
    X,Y = samples[:, :sliceLenOpt-1], samples[:,sliceLenOpt-1]
    Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=testPerc)
    svr.fit(Xtr, Ytr)
    print('Training error: '  , 1-svr.score(Xtr,Ytr))
    print('Validation error: ', 1-svr.score(Xva,Yva))
    np.append(minCerr, 1-svr.score(Xva,Yva))
print (minCerr)
print (minS)


import matplotlib.pyplot as pl
pl.figure()
pl.plot(Yva[:100], 'b')
pl.plot(svr.predict(Xva)[:100],'r')
pl.show()

pl.figure()
pl.plot(validation[:2000], 'b')
pl.show()