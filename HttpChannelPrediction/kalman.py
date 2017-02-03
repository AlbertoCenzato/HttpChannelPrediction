import numpy as np
from pykalman import KalmanFilter as KF

def kalmanFilter(params, Xva):
   size = len(params)
   A = np.zeros((size,size))
   for i in range(0,size-1):
      A[i,i+1] = 1
   A[size-1,:] = params
   
   C = np.zeros(size)
   C[0] = 1
   B = np.eye(size,size)
   kalman = KF(transition_matrices=A, observation_matrices=C, transition_covariance = B)
   
   numOfSamples = Xva.shape[0]
   predictions = np.zeros(numOfSamples)
   for t in range(0, numOfSamples):
      means, covariances = kalman.filter(Xva[t,:])

      next_mean, next_covariance = kalman.filter_update(means[-1], covariances[-1])
      predictions[t] = next_mean[-1]
   
   return predictions