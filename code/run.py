import implicit # see https://github.com/benfred/implicit
import pandas as pd
import numpy as np
import time
import scipy.sparse as sparse
from utils import load_from_SQL

def preprocess(df):
    '''Prepare data for modeling

    1. Create new 'item' column for (Origin City, Destination City) pairs. 
    2. Get user-item count to show number of times a user interacted with item. 
    3. Remove low-user trips and users with low-items. 
    4. Split to train-test sets. 
    5. Convert to sparse matrix for modeling.

    Returns
    ----------
    user_item:  
    '''
    print('\nPreprocessing ... ')
    df = _get_od_pairs(df)

    user_item_df = _get_user_item_count(df)

    user_item_df = _map_cat_to_numeric(user_item_df)

    user_item_mat = _convert_to_sparse_matrix(user_item_df)

    #user_item = _remove_low_freq_data(user_item)

    train, test_idx = _train_test_split(user_item)

    return user_item, train, test_idx

def _get_od_pairs(df):
    df.dropna(subset=['orgcity','destcity'],inplace=True)
    df['item'] = list(zip(df.orgcity,df.destcity))
    return df

def _get_user_item_count(df):
    '''Forms a df with number of interactions for each user and items.'''
    return df.groupby(['userid','item']).size().rename('freq').reset_index()
    #.unstack()

def _map_cat_to_numeric(user_item_df):
    '''Map "user" and "item" categories to unique numeric values.
    This is necessary for creating sparse matrix.'''
    user_item_df['userid'] = user_item_df['userid'].astype('category')
    user_item_df['item'] = user_item_df['item'].astype('category')
    return user_item_df

def _convert_to_sparse_matrix(user_item_df):
    return sparse.coo_matrix(user_item_df.freq,
                            (user_item_df.userid,user_item_df.item))

def _remove_low_freq_data(user_item):
    '''[LATER- Change this for coo_matrix]'''
    min_users_per_item = 2
    user_item = user_item.loc[:,user_item.count()>=min_users_per_item]

    # Note: data already meets requirement. See print(f'Items per user: {user_item.count(axis=1).agg(["min","mean","median"])}')
    min_items_per_user = 10
    user_item = user_item.loc[user_item.count(axis=1)>=min_items_per_user,:]
    return user_item

def _train_test_split(user_item_mat):
    '''[Change this for coo_matrix]
    Split data into train and test sets, using "leave-one-out" approach.

    Returns
    ----------
    train_mat: 
        user_item matrix with test values removed. 
    test_idx:
        array of indices for test values in original user_item matrix.'''
    test_idx = np.zeros(len(user_item.shape[0]))    # initialize array
    test_size = 1
    train_mat = user_item.copy()

    for i,user in user_item.iteritems():
        item_idx = user.notna().index
        test_item = np.random.choice(item_idx,test_size)
        test_idx[i] = test_item
        train_mat.loc[i,test_item] = np.nan
    return train_mat, test_idx

def validate_CF(model,N=5):
    '''Measure Recall accuracy using "leave-one-out" approach.

    Algorithm
    ----------
    1. For each user, move one item to test set (no duplicates should remain in train set).
    2. Get top N recommendations for each user.
    3. Measure the frequency that the left-out item is captured by top N recommendations.'''
    pass

def main():
    alpha = 15

    #df = load_from_SQL('dr250_active')
    df = pd.read_csv('data/dr250_active.csv')

    df = preprocess(df)
    input('Pause')

    #model = implicit.als.AlternatingLeastSquares
    #model.fit(mat)
    #score = validate_CF(model)

if __name__=='__main__':
    main()