import numpy as np 
import pandas as pd 


class AnomalyDetection:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon

    def fit(self, data):
        '''
        function returns mean and variance vectors both have the same size as
        features vector
        '''
        self.mu = np.float32(np.mean(data, axis=0))
        self.sigma2 = np.float32(np.var(data, axis=0, ddof=1))
    
    def density_estimation(self, data, mu, sigma2):
        '''
        function returns the probability of each data point using density 
        estimation algorithm
        '''
        p = []
        n = data.shape[1]
        for point in data:
            temp = 1
            for idx in range(n):
                temp *= (1 / np.sqrt(2 * np.pi * sigma2[idx])) * \
                        np.exp(-((point[idx] - mu[idx])**2 / (2*sigma2[idx])))
            p.append(temp)
        return np.array(p)

    def select_threshold(self, pval, yval):
        '''
        function selects best threshold value
        '''
        step_size = (np.max(pval) - np.min(pval)) / 1000
        self.bestF1 = 0
        for epsilon in np.arange(np.min(pval), np.max(pval), step_size):
            prediction = np.where(pval < epsilon, 1, 0)  
            tp = np.sum([(x == 1) & (y == 1) for x, y in zip(prediction, yval)])
            fp = np.sum([(x == 1) & (y == 0) for x, y in zip(prediction, yval)])
            fn = np.sum([(x == 0) & (y == 1) for x, y in zip(prediction, yval)])
            if (tp + fp) > 0:
                prec = (tp) / (tp + fp)
                rec = (tp) / (tp + fn)
                F1 = (2 * prec * rec) / (prec + rec)
            else:
                prec = 0
                rec = 0
                F1 = 0
                
            if F1 > self.bestF1:
                self.bestF1 = F1
                self.epsilon = epsilon

    def predict(self, data):
        '''
        function predicts anomalies in data 
        '''
        p = self.density_estimation(data, self.mu, self.sigma2)
        predictions = np.where(p < self.epsilon, 1, 0)  
        return predictions
            
    
# load data 
data = pd.read_csv('data.csv', header=None)
X = np.array(data)
xval = pd.read_csv('xval.csv', header=None)
yval = pd.read_csv('yval.csv', header=None)
xval = np.array(xval)
yval = np.array(yval)
# create instance of the class and evaluate results 
detector = AnomalyDetection()
detector.fit(X)
pval = detector.density_estimation(xval, detector.mu, detector.sigma2)
detector.select_threshold(pval, yval)
print(detector.epsilon, detector.bestF1)
p = detector.predict(X)
print(p[:10])
