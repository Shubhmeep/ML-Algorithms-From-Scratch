''' Import Libraries'''
import pandas as pd
import numpy as np


class Discriminant:
    ''' Prototype class for Discriminants'''
    def __init__(self):
        self.params = {}
        self.name = ''
        
    def fit(self, data):
        raise NotImplementedError
    
    def calc_discriminant(self, x):
        raise NotImplementedError

class GaussianDiscriminant(Discriminant):
    ''' Assumes a Gaussian Distribution for P(x|C_i)'''
    def __init__(self, data = None, prior=0.5, name = 'Not Defined'):
        '''Initialize pi and model parameters'''
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
    
    def fit(self, data):
        ''' Data is a numpy array consisting of data from a single class, where each row is a sample'''
        self.params['mu']    = np.mean(data)
        self.params['sigma'] = np.std(data)
        
        
    def calc_discriminant(self, x):
        '''Returns a discriminant value for a single sample'''
        mu = self.params['mu']
        sigma= self.params['sigma']
        prior = self.params['prior']
       
        '''Your code here'''
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive.")

        g_x = np.log(prior) - 0.5 * (np.log(2*self.pi*(sigma**2))) - 0.5 * ((x - mu)**2 / (sigma**2)) 
        return g_x

''' Create our MV Discriminant Class'''
class MultivariateGaussian(Discriminant):
    
    def __init__(self, data=None, prior=0.5, name = 'Not Defined'):
        '''Initialize pi and model parameters'''
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
        
    def fit(self, data):
        ''' Data is a numpy array consisting of data from a single class, where each row is a sample'''
        self.params['mu']    = np.average(data, axis=0)
        self.params['sigma'] = np.cov(data.T, bias=True)
        
    def calc_discriminant(self, x):
        mu, sigma, prior = self.params['mu'], self.params['sigma'], self.params['prior']
        '''Your code here'''

        d = sigma.shape[0]
        g_x = np.log(prior) - 0.5 * np.log((2 * np.pi)**d * np.linalg.det(sigma)) - 0.5 * np.dot((x - mu).T, np.dot(np.linalg.inv(sigma), (x - mu)))
        return g_x


        
        
        
