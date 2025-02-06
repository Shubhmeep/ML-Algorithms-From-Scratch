''' Import Libraries'''
import pandas as pd
import numpy as np
from discriminants import GaussianDiscriminant, MultivariateGaussian

class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError

class Priors(Classifier):
    ''' A classifier that uses only priors to determine what to output '''
    def __init__(self):
        super().__init__()
        self.class_priors = {}  
        
    def predict(self, x):
        if not self.class_priors:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        most_probable_class = None
        highest_prob = 0
        # Iterate over each class in the dictionary
        for cls in self.class_priors:
            prob = self.class_priors[cls]
            if prob > highest_prob:
                highest_prob = prob
                most_probable_class = cls
        return most_probable_class
    
    def fit(self, dataframe, label_key='Labels'):
        total_samples = 0
        for _ in dataframe[label_key]:
            total_samples += 1
        class_counts = {}
        for label in dataframe[label_key]:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1   
        self.class_priors = {}
        for label in class_counts:
            count = class_counts[label]
            prior = count / total_samples
            self.class_priors[label] = prior

''' Create our Discriminant Classifier Class'''    
class DiscriminantClassifier(Classifier):
    ''' A classifier that uses discriminant functions (one per class) for classification. '''
    def __init__(self):
        ''' Initialize Class Dictionary '''
        self.model_params = {}
        self.classes = {}      
        self.class_counts = {}
        
    def set_classes(self, *discs):
        '''Pass discriminant objects and store them in self.classes.
           This class is useful when you have existing discriminant objects'''
        for i in discs:
            self.classes[i.name] = i
        return self

    def fit(self, dataframe, label_key=['Labels'], default_disc=MultivariateGaussian):
        label_col = label_key[0]
        unique_labels = []  
        for lab in dataframe[label_col]:
            if lab not in unique_labels:
                unique_labels.append(lab)
        
        for lab in unique_labels:
            rows = []  
            for index, row in dataframe.iterrows():
                if row[label_col] == lab:
                    feature_row = []  
                    for col in dataframe.columns:
                        if col != label_col:
                            feature_row.append(row[col])
                    rows.append(feature_row)
            class_data = np.array(rows)
            
            disc = default_disc(data=class_data, prior=1.0)
            disc.name = lab
            self.classes[lab] = disc
            self.class_counts[lab] = class_data.shape[0]
        
        priors_obj = Priors()
        priors_obj.fit(dataframe, label_key=label_col)
 
        for lab in self.classes:
            self.classes[lab].params['prior'] = priors_obj.class_priors[lab]

    def predict(self, x):
        ''' 
        Returns a key (class label) corresponding to the highest discriminant value.
        If x is a single sample (1D array), it is reshaped into a 2D array with one row.
        '''
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        predictions = []
        for i in range(x.shape[0]):
            sample = x[i]
            best_label = None
            best_value = -1e99 
            for lab in self.classes:
                disc_val = self.classes[lab].calc_discriminant(sample)
                if disc_val > best_value:
                    best_value = disc_val
                    best_label = lab
            predictions.append(best_label)
        return np.array(predictions)

    def pool_variances(self):
  
        pooled_cov_num = None  
        pooled_cov_den = 0     

        for lab in self.classes:
            n_i = self.class_counts[lab]
            sigma_i = self.classes[lab].params['sigma']
            weighted_sigma = (n_i - 1) * sigma_i
            if pooled_cov_num is None:
                pooled_cov_num = weighted_sigma
            else:
                pooled_cov_num = pooled_cov_num + weighted_sigma
            pooled_cov_den = pooled_cov_den + (n_i - 1)
        
        pooled_cov = pooled_cov_num / pooled_cov_den
        
        for lab in self.classes:
            self.classes[lab].params['sigma'] = pooled_cov
        
        self.model_params['pooled_cov'] = pooled_cov
        return pooled_cov
