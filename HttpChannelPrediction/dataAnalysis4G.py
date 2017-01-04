import numpy as np

from utils import dataLoaderMat, dataLoaderVec, slicer
from crossValidation import chooseBestSplitLen
from sklearn.cross_validation import train_test_split

 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

data, N = dataLoaderVec('tram')

from sklearn import linear_model as lm

linReg = lm.LinearRegression()
len = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)
samples = slicer(data, len, True)
X,Y = samples[:, :len-1], samples[:,len-1]
Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.66)
linReg.fit(Xtr, Ytr)
print('optimal split size' , len)
print('Training error linReg Trams: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Trams: ', 1-linReg.score(Xva,Yva))

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
len = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)
samples = slicer(data, len, True)
X,Y = samples[:, :len-1], samples[:,len-1]
Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.66)
linReg.fit(Xtr, Ytr)
print('optimal split size' , len)
print('Training error linReg Cars: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Cars: ', 1-linReg.score(Xva,Yva))


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
'''

minCerr = []
minS = []
for c in [0.001, 0.005,0.01]:
    print('-----------------ITERAZIONE CON C = ', c, '----------------')
    svr = svm.LinearSVR(verbose = True, C = c, max_iter = 100000)
    sliceLenOpt = chooseBestSplitLen(data, svr, max_size = 10)
    minS = np.append(minS, sliceLenOpt)
    samples = slicer(data, sliceLenOpt, True)
    X,Y = samples[:, :sliceLenOpt-1], samples[:,sliceLenOpt-1]
    Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=testPerc)
    svr.fit(Xtr, Ytr)
    print('Training error SVM Trams: '  , 1-svr.score(Xtr,Ytr))
    print('Validation error SVM Trams: ', 1-svr.score(Xva,Yva))
    minCerr = np.append(minCerr, 1-svr.score(Xva,Yva))
print (minCerr)
print (minS)

data, N = dataLoaderVec('car')
minCerr = []
minS = []
for c in [0.001, 0.005,0.01]:
    print('-----------------ITERAZIONE CON C = ', c, '----------------')
    svr = svm.LinearSVR(verbose = True, C = c, max_iter = 100000)
    sliceLenOpt = chooseBestSplitLen(data, svr, max_size = 10)
    minS = np.append(minS, sliceLenOpt)
    samples = slicer(data, sliceLenOpt, True)
    X,Y = samples[:, :sliceLenOpt-1], samples[:,sliceLenOpt-1]
    Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=testPerc)
    svr.fit(Xtr, Ytr)
    print('Training error SVM Cars: '  , 1-svr.score(Xtr,Ytr))
    print('Validation error SVM cars: ', 1-svr.score(Xva,Yva))
    minCerr = np.append(minCerr, 1-svr.score(Xva,Yva))
print (minCerr)
print (minS)

'''
#SVM altro package
import sklearn.model_selection as m
data, N = dataLoaderVec('tram')
parameters = {'C':[0.001,0.01, 0.1, 1, 10], 'epsilon':[0.05, 0.1, 0.2, 0.3]}
svr = svm.LinearSVR()
clf = m.GridSearchCV(svr, parameters, verbose = 3)
samples = slicer(data, 6, True)
X,Y = samples[:, :6-1], samples[:,6-1]
Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=testPerc)
clf.fit(Xtr, Ytr)
print('Training error optimal SVM tram: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM tram: ', 1-clf.score(Xva,Yva))

'''
import matplotlib.pyplot as pl
pl.figure()
pl.plot(Yva[:100], 'b')
pl.plot(svr.predict(Xva)[:100],'r')
pl.show()

pl.figure()
pl.plot(validation[:2000], 'b')
pl.show()
'''