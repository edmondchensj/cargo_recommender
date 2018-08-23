import pandas as pd
from sqlalchemy import create_engine

def load_from_SQL(filename):
    print('* Loading data from PostgresSQL ...')
    engine = create_engine('postgresql://kenanzhang:knZ@NU@129.105.86.117:5432/truck_alliance')
    df = pd.read_sql_query(f'SELECT * FROM {filename} ORDER BY userid, datetime', engine)
    pd.set_option('max_colwidth',5)
    print(df.head())
    return df