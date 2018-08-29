import implicit # see https://github.com/benfred/implicit
import pandas as pd
import numpy as np
from preprocess import (filter_actions,get_user_items,train_test_split)
from validate import (validate,validate_pct_rank,plot_pct_rank)

'''Runs model for specific k and N values and other parameters.'''
def main():
    # Load Data
    df = pd.read_csv('data/dr250_active.csv')

    # Preprocess Data
    df = filter_actions(df)
    user_items = get_user_items(df)
    k = 20
    user_items_train, test_users = train_test_split(user_items,
                                                    split_method='leave_k_out',
                                                    k=k)

    # Build Model (see https://github.com/benfred/implicit)
    model = implicit.als.AlternatingLeastSquares(factors=10,
                                    regularization=0.1,
                                    iterations=30)
    alpha=2
    item_users_train = user_items_train.transpose()
    model.fit(item_users_train*alpha)

    # Validate Model
    N = 100
    mean_model_acc, mean_benchmark_acc = validate(model,user_items,user_items_train,test_users,N=N)

if __name__=='__main__':
    main()