import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse

def filter_actions(df,acttypes=['click','call']):
    return df[df.acttype.isin(acttypes)]
    
def get_user_items(df):
    '''Prepares a user-item matrix for modeling

    1. Creates new 'item' column for (Origin City, Destination City) pairs. 
    2. Gets user-item count to show number of times a user interacted with item. 
    3. Converts to sparse matrix for modeling.

    Returns
    ----------
    user_items: sparse matrix
        Matrix with rows:users, columns:items. Each cell holds the number of interactions.
    '''
    df = _get_od_pairs(df)
    user_item_df = _get_user_item_count(df)
    user_item_df = _map_cat_to_numeric(user_item_df)
    user_items = _convert_to_sparse_matrix(user_item_df)
    return user_items

def _get_od_pairs(df):
    df = df.copy()
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

def train_test_split(user_items,
                    split_method='leave_k_out',
                    k=10,test_ratio=0.2,
                    sample_size=0.05):
    '''Splits data into train and test sets.
    Run random.seed(0) externally for reproducability.

    Parameters
    ----------
    user_items: sparse matrix
        Full user-item data.
    split_method: str, optional
        Options: 'leave_k_out','random_sampling'
    k: int, optional
        For split_method=='leave_k_out' only.
        k number of items will be moved to test set.
    test_ratio: float, optional
        For split_method=='leave_k_out' only.
        Provides fraction of users to move to test set.
    sample_size: float, optional
        For split_method=='random_sampling' only.
        Provides fraction of data to move to test set.

    Returns
    ----------
    user_items_train: sparse matrix
        User-item data for train set (test values removed). 
    test_users: 1d array
        Row indices of test users in user_items.
    '''
    user_items = user_items.tocsr()
    user_items_train = user_items.copy()

    if split_method=='leave_k_out':
        '''Procedure: 
        1. Split users into train and test. 
            (Users in test set must have at least k*10 unique items.)
        2. For users in test set, leave k items out. 
        '''
        # Split users into train and test
        num_users = user_items.shape[0]
        num_test_users = int(np.ceil(test_ratio*num_users))
        items_per_user = np.diff(user_items.indptr)
        eligible_test_users = [user for user,val in enumerate(items_per_user) 
                                if items_per_user[user] > k*10]
        if len(eligible_test_users) < num_test_users:
            raise ValueError('Not enough test users to meet requirement.'
                            'Please reduce k value or increase test_ratio.')
        test_users = np.sort(random.sample(eligible_test_users,num_test_users))

        # Remove k test items from train set
        for user in test_users:
            items = list(user_items[user,:].nonzero()[1])
            k_items = random.sample(items,k)
            user_items_train[user,k_items] = 0

    elif split_method=='random_sampling':
        '''Randomly samples a portion of data.'''
        # Get list of (user_idx,item_idx) where frequency is non-zero
        user_item_lst = list(zip(user_items.nonzero()[0],
                                    user_items.nonzero()[1]))
        # Get test indices
        test_size = int(np.ceil(sample_size*len(user_item_lst)))
        test_indices = random.sample(user_item_lst,test_size)
        test_users, test_items = zip(*test_indices)

        # Remove test values from train set
        user_items_train[test_users,test_items] = 0     

    user_items_train.eliminate_zeros()
    user_items_train.tocoo()
    return user_items_train, test_users