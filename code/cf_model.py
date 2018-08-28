import pandas as pd
import numpy as np
import time
from .utils import load_from_SQL, timer

class WeightedRegularizedMF(object):
    '''Weighted Regularized Matrix Factorization (WRMF) for Implicit Feedback Dataset

    WRMF is a Collaborative Filtering model customized for truck-alliance's "active" data. 
    Matrix Factorization is a model-based approach and is currently used in state-of-the-art recommender systems (He et al 2017). 
    It is different from a basic Matrix Factorization (MF) model in that it gives greater weightage to frequent trips. 

    Technical Details
    ----------
    WRMF is based on the paper "Collaborative Filtering for Implicit Feedback Datasets" by Hu et al and ... (?)
    It uses a latent-factor model approach, which is advantageous over memory-based approaches for its scalability.

    '''
    pass

class MemoryBasedCF(object):
    ''' Memory Based Collaborative Filtering (UbCF) for Implicit Feedback Dataset

    Collaborative filtering generally has two approaches: memory/neighborhood-based and model-based. 
    While memory-based is simpler to implement, they are less scalable and generally outperformed by model-based CF. 

    Algorithm (memory-based)
    ----------
    (I) Preprocess
        (a) Retain only acttype "click" and "call".
        (b) Split each user's data into train and test sets using a cutoff ratio. (* could set a time cutoff.)

    (II) Training
    For each driver in train set, 
        (a) Find top N most similar drivers. Similarity based on: 
                User-user
                (i) Origin-Destination (OD) city pairs (user-user)
                (ii) Name and frequency of home city, determined by user's most frequented city. 
                (iii) Time of "Order" (try later) 

                Item-item? Only feature is origin/dest. 
        (b) Then, from top N drivers, find top k OD pairs not in the driver's train set.

    (III) Validation
    Detailed-approach:
    We will use the evaluation approach described by Hu et al.(2008).

    Naive-approach:
    For each driver, 
        Determine percentage of new OD pairs in test set that are in top k predictions. 
        (* could add predictive confidence metric?)
    '''
    def __init__(self,N=10,k=10):
        self.N = N
        self.k = k

        # User-item matrix where items are all unique OD pairs found in the dataset. 
        self.matrix = None

        # User-based similarities

        # Attributes below are each a Series with drivers as index. 
        self.home_city = None 
        self.top_n_drivers = None 
        self.top_k_preds = None
        self.od_train = None
        self.od_test = None
        self.od_unseen = None 

    @timer
    def fit(self,df,split_ratio=0.8,subsample=False,ss_num_users=None):
        '''Trains CF model, tests predictions, and prints the model accuracy. 

        Parameters 
        ----------
        data : DataFrame
            Raw "active" cargo data. Contains columns 'userid','acttype','orgcity','destcity','ordercreattime'.
        split_ratio : float, optional
            Use this ratio to split each driver's data to train and test sets. 
        subsample: bool, optional
            Train CF model for each driver based on a random subset of other drivers, to reduce computation time. 
        ss_num_users: int, optional
            Number of drivers to subsample if subsample is set to True. 
        '''
        train_df, test_df = preprocess(df,split_ratio)

        get_features(train_df,test_df)

        find_N_similar_drivers(train_df)

        find_k_preds()

        test_preds()
        
        get_score()

    def preprocess(self,df,split_ratio):
        '''Retains only acttype "click" or "call" and splits data to train and test sets.'''
        df = df[df.acttype.isin(["click","call"])]

        df['ordercreattime'] = pd.to_datetime(df.ordercreattime)
        df.sort_values(by=['userid','datetime'],inplace=True)

        train_df = df.groupby('userid').apply(lambda user: user.head(split_ratio*user.shape[0]))
        test_df = df.groupby('userid').apply(lambda user: user.tail((1-split_ratio)*user.shape[0]))
        return train_df, test_df

    def get_features(self,train_df,test_df):
        '''Gets OD pairs and home city (and time?) for each driver.'''
        _get_OD_pairs(train_df,test_df)
        _get_home_city(train_df)
        #_get_order_time(df)

    def _get_OD_pairs(self,train_df,test_df):
        self.od_train = train_df.groupby('userid').apply(lambda user: list(zip(user.orgcity,user.destcity)),axis=1)
        self.od_test = test_df.groupby('userid').apply(lambda user: list(zip(user.orgcity,user.destcity)),axis=1)
        self.od_unseen = list(self.od_train - self.od_test)

    def _get_home_city(self,train_df):
        cities = train_df[['orgcity','destcity']]
        city_count = cities.groupby('userid').apply(pd.Series.value_counts().rename('city'))
        city_count['total'] = city_count.sum(axis=1)
        print(city_count.head()) # need to know column headers.
        self.home_city = city_count.groupby('userid').head(1)

    def find_N_similar_drivers(self):
        '''Finds top N similar drivers of each driver. 
        Similarity based on OD city pairs and home city (and time?).'''
        _find_by_od

    def _find_by_od(self):
        top_drivers_by_od = self.od_train.apply()



    def find_k_preds(self):
        pass

def main():
    df = load_from_SQL('dr250_active')
    cf = UserBasedCF()

    cf.fit(df)

if __name__ == '__main__':
    #main()