import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sqlalchemy import create_engine
import time

def set_plt_style():
    plt.style.use('ggplot')
    mpl.rcParams['figure.dpi'] = 200
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'
    plt.rcParams['legend.fontsize'] = 'small'

def histogram(data,xlabel,ylabel='Count',title=None,hist_max=20,bins=100,figscale=1):
    fig = plt.figure(figsize=(7*figscale,3*figscale))
    ax = plt.axes()
    plt.hist(data,bins=bins,range=(data.min(),hist_max))
    ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def load_from_SQL(filename):
    print('* Loading data from PostgresSQL ...')
    engine = create_engine('postgresql://kenanzhang:knZ@NU@129.105.86.117:5432/truck_alliance')
    df = pd.read_sql_query(f'SELECT * FROM {filename} ORDER BY userid, datetime', engine)
    pd.set_option('max_colwidth',5)
    print(df.head())
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