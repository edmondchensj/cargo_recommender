import pandas as pd
from sqlalchemy import create_engine
import time

def load_from_SQL(filename):
    print('* Loading data from PostgresSQL ...')
    engine = create_engine('postgresql://kenanzhang:knZ@NU@129.105.86.117:5432/truck_alliance')
    df = pd.read_sql_query(f'SELECT * FROM {filename} ORDER BY userid, datetime', engine)
    pd.set_option('max_colwidth',5)
    print(df.head())
    return df

def train_test_split(df,split_ratio=0.8):
    '''Splits data into train and test sets using ratios. '''
    train_df = df.groupby('userid').apply(lambda user: user.head(split_ratio*user.shape[0]))
    test_df = df.groupby('userid').apply(lambda user: user.tail((1-split_ratio)*user.shape[0]))
    return train_df, test_df

def retain_acttype(df):
    # Retain only acttype "click" or "call" (for dr250 only). 
    df = df[df.acttype.isin(["click","call"])].reset_index(drop=True)
    return df

def timer(f,subfunc=False): # function timer
    def timed(*args,**kw):
        ts = time.time()
        ret = f(*args,**kw)
        te = time.time()
        if subfunc: 
            msg = f'{f.__name__} took {te-ts:.2f}s.'
        else:
            msg = f'Total time elapsed: {te-ts:.2f}s.'
        print(msg)
        return ret
    return timed