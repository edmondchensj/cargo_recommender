import pandas as pd
import numpy as np
import time
from .utils import load_from_SQL

class UserBasedCF(object):
    ''' User-based Collaborative Filtering (UbCF)

    A recommender system based on UbCF, customized for truck-alliance's "active" data.

    Algorithm
    ----------
    (I) Training
    For each driver in data, 
        (a) Find top N most similar drivers. Similarity based on: 
                Origin-Destination (OD) pairs under action type "click" or "call"
                Home city
                Time of Action (try later) 
        (b) Then, find top k OD pairs from top N drivers that are not in train set.

    (II) Validation
    For each driver in data, 
        Determine percentage of new OD pairs in test set that are in top k predictions. 
    '''
    def __init__(self,N=10,k=10):
        self.N = N
        self.k = k

        self.od = None # a Series with list of OD city pairs for each driver. 
        self.home_city = None # a Series with home city of each driver. 
        self.top_n_drivers = None # a Series with list of topn similar drivers for each driver. 
        self.top_k_preds = None  # a Series with list of topk OD predictions for each driver. 
        self.od_unseen = None # a Series with list of unseen (in train set) OD pairs for each driver. 

    def fit(data,train_test_split=0.8,subsample=False,ss_num_users=None):
        '''Trains CF model, tests predictions, and returns the model accuracy. 

        Parameters 
        ----------
        data : DataFrame
            Raw "active" cargo data. Contains columns 'userid','orgcity','destcity','ordercreattime'.
        train_test_split : float, optional
            Split each driver's data to train and test sets. 
        subsample: bool, optional
            Train CF model for each driver based on a random subset of other drivers, to reduce computation time. 
        ss_num_users: int, optional
            Number of drivers to subsample if subsample is set to True. 
        '''
        get_features(data)

        find_similar_drivers()

        find_k_preds()

        test_preds()
        
        get_score()

    def get_features(data):
        '''Gets OD pairs and home city (and time?) for each driver.'''
        self.od = ...
        self.home_city = ...
        self.od_unseen = ...

    def find_similar_drivers():
        '''Finds top N similar drivers of each driver. 
        Similarity based on OD city pairs and home city (and time?).'''
        pass

    def find_k_preds():
        pass

def main():
    df = load_from_SQL('dr250_active')
    cf = UserBasedCF()
    cf.fit(df)

if __name__ == '__main__':
    #main()