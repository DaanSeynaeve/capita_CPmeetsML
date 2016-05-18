'''
Created on May 11, 2016

@author: Daan Seynaeve
'''
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import numpy as np

class AdvancedModel():
    
    clusters = []
    
    # price class regression
    price_reg = LinearRegression()
        
    def fit(self, X_train, y_train, n_clusters=4):
        y_train_mat = np.array(y_train).reshape((-1,1))
        
        # 1. determine clusters
        self.km = KMeans(n_clusters=5)
        self.km.fit(y_train_mat)
        clusters = self.km.cluster_centers_
        cluster_indices = self.km.predict(y_train_mat)
        print(clusters)
        
        # 2. fit naive bayes
        #self.nb.fit(X_train, ...)
        #self
        
        # 3. train regression model
        #price_reg.fit
        
    def predict(self, X):
        pass
        
    def get_weights(self):
        return np.append(self.price_reg.coef_, [self.price_reg.intercept_])
        
    def set_weights(self, w):
        self.price_reg.coef_ = w[:-1]
        self.price_reg.intercept_ = w[-1]
        