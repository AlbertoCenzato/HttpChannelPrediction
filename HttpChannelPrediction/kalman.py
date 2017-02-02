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

      # questa riga d� problemi. lancia un'eccezione: setting an array element with a sequence.
      # dalla documentazione (https://pykalman.github.io/#pykalman.KalmanFilter.filter_update)
      # pare che filter_update() restituisca un array che dovrebbe essere "mean estimate for 
      # state at time t+1 given observations from times [1...t+1]" quindi nel nostro caso dovrebbe
      # essere un array di dimensione 1, ma è una matrice 8x8!
      next_mean, next_covariance = kalman.filter_update(means[-1], covariances[-1])
      predictions[t] = next_mean[-1]
   
   return predictions