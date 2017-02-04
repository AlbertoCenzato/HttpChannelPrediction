import numpy as np

from utils import dataLoaderMat, dataLoaderVec, slicer
from crossValidation import chooseBestSplitLen
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model as lm

 #-------------------------------------------------
 #--------- LINEAR REGRESSION ---------------------
 #-------------------------------------------------

 #------ tram ------
print('----- PREDICTION FOR TRAMS -----')
data, N = dataLoaderVec('tram')

# choose optimal split size (model dimension)
linReg = lm.LinearRegression()
tram_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)

# training with optimal split size
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 
linReg.fit(Xtr, Ytr)
print('optimal split size' , tram_opt_dim)
print('Training error linReg Trams: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Trams: ', 1-linReg.score(Xva,Yva))


# ------ cars ------
print('----- PREDICTION FOR CARS -----')
data, N = dataLoaderVec('car')

# choose optimal split size (model dimension)
linReg = lm.LinearRegression()
car_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)

# training with optimal split size
samples = slicer(data, car_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 

linReg.fit(Xtr, Ytr)
print('optimal split size' , car_opt_dim)
print('Training error linReg Cars: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg Cars: ', 1-linReg.score(Xva,Yva))



# ------ trasporti urbani -----
print('----- PREDICTION FOR TRASPORTI URBANI -----')
data, N = dataLoaderVec('trasporti_urbani')
# choose optimal split size (model dimension)
linReg = lm.LinearRegression()
urb_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)

# training with optimal split size
samples = slicer(data, urb_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 

linReg.fit(Xtr, Ytr)
print('optimal split size' , urb_opt_dim)
print('Training error linReg trasporti urbani: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg trasporti urbani: ', 1-linReg.score(Xva,Yva))


# ------ cars and trains ------
print('----- PREDICTION FOR CARS AND TRAINS -----')
data, N = dataLoaderVec('cars_and_trains')

# choose optimal split size (model dimension)
linReg = lm.LinearRegression()
cat_opt_dim = chooseBestSplitLen(data, linReg, max_size = 10, plot = True)

# training with optimal split size
samples = slicer(data, cat_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=0.66)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 

linReg.fit(Xtr, Ytr)
print('optimal split size' , cat_opt_dim)
print('Training error linReg cars and trains: '  , 1-linReg.score(Xtr,Ytr))
print('Validation error linReg cars and trains: ', 1-linReg.score(Xva,Yva))


#-------------------------------------------
#------------ USING SVMs -------------------
#-------------------------------------------

print('\n------------------------------')
print('------- USING SVMs --------')
print('-------------------------------')

from sklearn import svm
testPerc = 0.66

# ------ tram ------
print('----- PREDICTION FOR TRAMS -----')
data, N = dataLoaderVec('tram')

# choose optimal split size (model dimension)
svr = svm.LinearSVR(C=0.03, epsilon=0.5)
tram_opt_dim = chooseBestSplitLen(data, svr, max_size = 10, plot = True)
'''
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 
'''
print('optimal split size' , tram_opt_dim)
#print('Training error SVM Trams: '  , 1-svr.score(Xtr,Ytr))
#print('Validation error SVM Trams: ', 1-svr.score(Xva,Yva))


# choose optimal model parameters
#tram_opt_dim = 5
parameters = {'C':[0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 'epsilon':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
svr = svm.LinearSVR()
clf = model_selection.GridSearchCV(svr, parameters, verbose = 0)
samples = slicer(data, tram_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva)
clf.fit(Xtr, Ytr)
print('Best params: ', clf.best_params_)
print('Training error optimal SVM tram: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM tram: ', 1-clf.score(Xva,Yva))

# training with optimal parameters
svr = clf.best_estimator_
svr.fit(Xtr, Ytr)
Yva_hat = svr.predict(Xva)

# plot sample prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(Yva_hat[1000:1080], label='Predicted value')
plt.plot(Yva[1000:1080], label='Measured value')
plt.ylabel('KB/s')
plt.show()





# ------ car ------
print('----- PREDICTION FOR CARS -----')
data, N = dataLoaderVec('car')

# choose optimal split size (model dimension)
svr = svm.LinearSVR(C=0.03, epsilon=0.5)
car_opt_dim = chooseBestSplitLen(data, svr, max_size = 10, plot = True)
'''
samples = slicer(data, car_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 
'''
print('optimal split size' , car_opt_dim)
#print('Training error SVM cars: '  , 1-svr.score(Xtr,Ytr))
#print('Validation error SVM cars: ', 1-svr.score(Xva,Yva))


# choose optimal model parameters
#car_opt_dim = 5
parameters = {'C':[0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 'epsilon':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
svr = svm.LinearSVR()
clf = model_selection.GridSearchCV(svr, parameters, verbose = 0)
samples = slicer(data, car_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva)
clf.fit(Xtr, Ytr)
print('Best params: ', clf.best_params_)
print('Training error optimal SVM cars: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM cars: ', 1-clf.score(Xva,Yva))

# training with optimal parameters
svr = clf.best_estimator_
svr.fit(Xtr, Ytr)
Yva_hat = svr.predict(Xva)

# plot sample prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(Yva_hat[1000:1080], label='Predicted value')
plt.plot(Yva[1000:1080], label='Measured value')
plt.ylabel('KB/s')
plt.show()


# ------ trasporti_urbani ------
print('----- PREDICTION FOR trasporti_urbani -----')
data, N = dataLoaderVec('trasporti_urbani')

# choose optimal split size (model dimension)
svr = svm.LinearSVR(C=0.03, epsilon=0.5)
trasporti_urbani_opt_dim = chooseBestSplitLen(data, svr, max_size = 10, plot = True)
'''
samples = slicer(data, trasporti_urbani_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 
'''
print('optimal split size' , trasporti_urbani_opt_dim)
#print('Training error SVM trasporti_urbani: '  , 1-svr.score(Xtr,Ytr))
#print('Validation error SVM trasporti_urbani: ', 1-svr.score(Xva,Yva))


# choose optimal model parameters
#trasporti_urbani_opt_dim = 5
parameters = {'C':[0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 'epsilon':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
svr = svm.LinearSVR()
clf = model_selection.GridSearchCV(svr, parameters, verbose = 0)
samples = slicer(data, trasporti_urbani_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva)
clf.fit(Xtr, Ytr)
print('Best params: ', clf.best_params_)
print('Training error optimal SVM trasporti_urbani: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM trasporti_urbani: ', 1-clf.score(Xva,Yva))

# training with optimal parameters
svr = clf.best_estimator_
svr.fit(Xtr, Ytr)
Yva_hat = svr.predict(Xva)

# plot sample prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(Yva_hat[1000:1080], label='Predicted value')
plt.plot(Yva[1000:1080], label='Measured value')
plt.ylabel('KB/s')
plt.show()


# ------ cars_and_trains ------
print('----- PREDICTION FOR cars_and_trains -----')
data, N = dataLoaderVec('cars_and_trains')

# choose optimal split size (model dimension)
svr = svm.LinearSVR(C=0.03, epsilon=0.5)
cars_and_trains_opt_dim = chooseBestSplitLen(data, svr, max_size = 10, plot = True)
'''
samples = slicer(data, cars_and_trains_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva) 
'''
print('optimal split size' , cars_and_trains_opt_dim)
#print('Training error SVM cars_and_trains: '  , 1-svr.score(Xtr,Ytr))
#print('Validation error SVM cars_and_trains: ', 1-svr.score(Xva,Yva))


# choose optimal model parameters
#cars_and_trains_opt_dim = 5
parameters = {'C':[0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 'epsilon':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
svr = svm.LinearSVR()
clf = model_selection.GridSearchCV(svr, parameters, verbose = 0)
samples = slicer(data, cars_and_trains_opt_dim, True)
X,Y = samples[:, :-1], samples[:,-1]
Xtr, Xva, Ytr, Yva = model_selection.train_test_split(X, Y, test_size=testPerc)
scaler = preprocessing.StandardScaler().fit(X)
Xtr = scaler.transform(Xtr)
Xva = scaler.transform(Xva)
clf.fit(Xtr, Ytr)
print('Best params: ', clf.best_params_)
print('Training error optimal SVM cars_and_trains: '  , 1-clf.score(Xtr,Ytr))
print('Validation error optimal SVM cars_and_trains: ', 1-clf.score(Xva,Yva))

# training with optimal parameters
svr = clf.best_estimator_
svr.fit(Xtr, Ytr)
Yva_hat = svr.predict(Xva)

# plot sample prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(Yva_hat[1000:1080], label='Predicted value')
plt.plot(Yva[1000:1080], label='Measured value')
plt.ylabel('KB/s')
plt.show()
'''
# ----- FILTRO DI KALMAN -----

from kalman import kalmanFilter

best_params = clf.best_estimator_.coef_
print('tram_opt_dim: ', tram_opt_dim)
predictions = kalmanFilter(best_params,Xva)


plt.figure()
plt.plot(predictions[1000:1100])
plt.plot(Yva[1000:1100])
plt.show()
'''