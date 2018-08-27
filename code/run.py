from implicit.als import AlternatingLeastSquares # see https://github.com/benfred/implicit
import pandas as pd
import numpy as np
import time
import random
import scipy.sparse as sparse
import pickle
from sklearn import metrics
from utils import load_from_SQL

# Blog explanations: 
# https://towardsdatascience.com/recommending-github-repositories-with-google-bigquery-and-the-implicit-library-e6cce666c77
# https://jessesw.com/Rec-System/
def preprocess(df):
    '''Prepares a user-item matrix for modeling

    1. Create new 'item' column for (Origin City, Destination City) pairs. 
    2. Get user-item count to show number of times a user interacted with item. 
    3. Convert to sparse matrix for modeling.

    Returns
    ----------
    user_items: sparse matrix
        a matrix with rows:users, columns:items. each cell holds the number of interactions.
    '''
    print('\nPreprocessing ... ')
    df = df[df.acttype.isin(['click','call'])]  # filter only acttype "click" and "call"
    df = _get_od_pairs(df)
    user_item_df = _get_user_item_count(df)
    user_item_df = _map_cat_to_numeric(user_item_df)
    user_items = _convert_to_sparse_matrix(user_item_df)
    return user_items

def _get_od_pairs(df):
    df.dropna(subset=['orgcity','destcity'],inplace=True)
    df['item'] = list(zip(df.orgcity,df.destcity))
    return df

def _get_user_item_count(df):
    '''Finds number of interactions for each user and items.'''
    user_item_df = df.groupby(['userid','item']).size().rename('freq').reset_index()
    return user_item_df

def _map_cat_to_numeric(user_item_df):
    '''Maps "user" and "item" categories to unique numeric values.
    This is necessary for creating sparse matrix.'''
    user_item_df['userid'] = user_item_df.userid.astype('category')
    user_item_df['item'] = user_item_df.item.astype('category')
    return user_item_df

def _convert_to_sparse_matrix(user_item_df):
    return sparse.coo_matrix((user_item_df.freq,
                            (user_item_df.userid.cat.codes,
                                user_item_df.item.cat.codes)))

def train_test_split(user_items,pct_test=0.05,split_method='random_sampling'):
    '''Splits data into train and test sets.

    Parameters
    ----------
    user_items: sparse matrix
        full user-item data.
    pct_test: float, optional
        percentage of data to move to test set.
    split_method: str, optional
        options: 'random_sampling' (may add other methods in future)

    Returns
    ----------
    user_items_train: sparse matrix
        user-item data for train set (test values removed). 
    test_indices: 1d array
        array of indices for test values in original user_item matrix.
    '''
    user_items = user_items.tocsr()
    if split_method=='random_sampling':
        # Get list of (user_idx,item_idx) where frequency is non-zero
        user_item_lst = list(zip(user_items.nonzero()[0],
                                    user_items.nonzero()[1]))
        # Get test indices
        random.seed(0) # for reproducability
        test_size = int(np.ceil(pct_test*len(user_item_lst)))
        test_indices = random.sample(user_item_lst,test_size)

    # Remove test values from train set
    user_items_train = user_items.copy()
    test_users, test_items = zip(*test_indices)
    user_items_train[test_users,test_items] = 0
    user_items_train.eliminate_zeros()
    user_items_train.tocoo()
    return user_items_train, test_users

def validate(model,user_items,user_items_train,test_users):
    '''Measures accuracy of model on test set.  

    Returns
    ----------
    accuracy: float
        percentage accuracy of correct recommendations
    '''
    scores = []
    user_items = user_items.tocsr()
    user_items_train = user_items_train.tocsr()

    for user in test_users:
        train_items = user_items_train[user,:].toarray().reshape(-1).nonzero()
        all_items = user_items[user,:].toarray().reshape(-1).nonzero()
        test_items = np.setdiff1d(all_items,train_items)

        num_test_items = len(test_items)
        recommendations = model.recommend(user,
                                        user_items_train,
                                        N=num_test_items*5)

        matched = np.intersect1d(recommendations,test_items)
        score = len(matched)/num_test_items
        scores.append(score)

        '''
        print(f'Number of train items: {len(train_items[0])}')
        print(f'Number of test items: {num_test_items}')
        print(f'Number of recommendations: {len(recommendations)}')
        print(f'Matched items: {matched}')
        '''

    accuracy = float(f'{np.mean(scores)*100:.3f}')
    return accuracy

def main():
    # Load Data]
    df = pd.read_csv('data/dr250_active.csv')

    # Prepare Data
    user_items = preprocess(df)
    user_items_train, test_users = train_test_split(user_items)

    '''
    # Build Model
    # Model based on paper by Hu et al. "Collaborative Filtering for Implicit Datasets"
    model = AlternatingLeastSquares(factors=10,
                                    regularization=0.01,
                                    iterations=50)
    confidence=15
    item_users_train = user_items_train.transpose()     # model fit takes in rows:items, columns:users
    model.fit(item_users_train*confidence)
    with open('data/model.pkl','wb') as f:
        pickle.dump(model,f)
    '''

    with open('data/model.pkl','rb') as f:
        model = pickle.load(f)
    # Validate Model
    accuracy = validate(model,user_items,user_items_train,test_users)
    print(f'Accuracy is: {accuracy}%')

if __name__=='__main__':
    main()