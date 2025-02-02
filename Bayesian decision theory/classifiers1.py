''' Import Libraries'''
import pandas as pd
import numpy as np
from discriminants import GaussianDiscriminant


class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError

class Prior(Classifier):
    
    def __init__(self):
        ''' Your code here '''
        self.model_params = {}
        pass
    

    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x (numpy array), y (numpy array)'''
        raise NotImplementedError



''' Create our Discriminant Classifier Class'''    
class DiscriminantClassifier(Classifier):
    ''''''
    def __init__(self):
        ''' Initialize Class Dictionary'''
        self.model_params = {}
        
    def set_classes(self, *discs):
        '''Pass discriminant objects and store them in self.classes
            This class is useful when you have existing discriminant objects'''
        raise NotImplementedError

            
    def fit(self, dataframe, label_key=['Labels'], default_disc=GaussianDiscriminant):
        ''' Calculates model parameters from a dataframe for each discriminant.
            Label_Key specifies the column that contains the class labels. ''' 
        raise NotImplementedError

                
    
    def predict(self, x):
        ''' Returns a Key (class) that corresponds to the highest discriminant value'''

        raise NotImplementedError

    def pool_variances(self):
        ''' Calculates a pooled variance and sets the corresponding model params '''

        raise NotImplementedError        

       
        
        
        
