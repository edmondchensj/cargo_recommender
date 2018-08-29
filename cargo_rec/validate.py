import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def validate(model,user_items,user_items_train,test_users,N=50,k=20,show_result=True):
    '''Given a specific N, measures recall accuracy of the model and a benchmark.
    
    Returns
    ----------
    mean_model_accuracy: float
        Recall accuracy of correct predictions using top N recommendations for each user. Averaged across all users.
        Formula: 
            (# items in both predictions and test set) / (# items in test set)
    
    mean_benchmark_accuracy: float
        Recall accuracy of correct predictions using the N most popular items in general. Averaged across all users.
        Same formula as above.
    '''
    model_accuracy, benchmark_accuracy = _create_empty_arrays(len(test_users),2)
    user_items = user_items.tocsr()
    user_items_train = user_items_train.tocsr()
    N_most_popular = _most_popular_items(user_items,N)

    for i,user in enumerate(test_users):
        all_items = user_items[user,:].toarray().reshape(-1).nonzero()
        train_items = user_items_train[user,:].toarray().reshape(-1).nonzero()
        test_items = np.setdiff1d(all_items,train_items)

        predictions = model.recommend(user,
                                    user_items_train,
                                    N=N)

        model_accuracy[i] = _recall_accuracy(predictions,test_items)
        benchmark_accuracy[i] = _recall_accuracy(N_most_popular,test_items)

    mean_model_acc = float(f'{np.mean(model_accuracy):.3f}')
    mean_benchmark_acc = float(f'{np.mean(benchmark_accuracy):.3f}')
    if show_result:
        _show_val_result(mean_model_acc,mean_benchmark_acc,k,N)
    else:
        return mean_model_acc,mean_benchmark_acc

def _show_val_result(mean_model_accuracy,mean_benchmark_accuracy,k,N):
    print('\nAccuracy (Recall)\n'
            '----------'
            f'\nModel: {mean_model_accuracy}%'
            f'\nBenchmark: {mean_benchmark_accuracy}%' 
            f'\n\n[Parameters: k={k}, N={N}]')
    
def _most_popular_items(user_items,N):
    item_counts = np.array(user_items.sum(axis=0)).reshape(-1)
    topN_items = (-item_counts).argsort()[:N]
    return topN_items

def _recall_accuracy(predictions,test_items):
    correct_predictions = np.intersect1d(predictions,test_items)
    return len(correct_predictions)/len(test_items)*100

def validate_pct_rank(model,user_items,user_items_train,test_users,
                    min_pct=0,max_pct=5,num_val=20):
    '''Given percentile parameters, measures recall accuracy of model and benchmark. 

    Returns
    ----------
    pcts: 1d array
        Percentiles given by N / # items * 100. 
    model_acc: 1d array
        Model recall accuracy for each percentile.
    benchmark_acc: 1d array
        Benchmark recall accuracy for each percentile. 
    '''
    num_items = user_items.shape[1]
    N_values = _get_N_values(num_items,min_pct,max_pct,num_val)
    pcts, model_acc, benchmark_acc = _create_empty_arrays(len(N_values),3)

    for i,N in enumerate(N_values):
        model_acc[i],benchmark_acc[i] = validate(model,
                                                user_items,
                                                user_items_train,
                                                test_users,
                                                N=N,
                                                show_result=False)
        pcts[i] = N/num_items*100
    return pcts, model_acc, benchmark_acc

def _get_N_values(num_items,min_pct,max_pct,num_val):
    start_N = int(np.ceil(min_pct/100*num_items))
    stop_N = int(np.ceil(max_pct/100*num_items))
    step = int(np.ceil((stop_N - start_N)/num_val))
    return range(start_N, stop_N, step)

def _create_empty_arrays(size,n):
    for i in range(n):
        yield np.zeros(size)

def plot_pct_rank(pcts,model_acc,benchmark_acc,num_items,k):
    '''Plots percentile-ranking graph using output of validate_pct_rank'''
    fig = plt.figure()
    ax = fig.add_axes((0,0.3,0.8,0.6))
    ax.plot(pcts,model_acc,'r',label='Model')
    ax.plot(pcts,benchmark_acc,'b',label='Benchmark')
    ax.legend()
    ax.set_title(f'Recommender Accuracy for k={k} test items')
    ax.set_ylabel('Recall (%)')
    ax.set_xlabel('Percentile (%)')
    xticks = ax.get_xticks()
    xlabels_N = _pct_to_N(xticks,num_items)
    ax2 = _add_new_xlabel(fig,pcts,y=.17,xticks=xticks,xlabels=xlabels_N,label='N')
    xlabels_Nk = _N_to_Nk(xlabels_N,k)
    ax3 = _add_new_xlabel(fig,pcts,y=.05,xticks=xticks,xlabels=xlabels_Nk,label='N/k')
    plt.show()

def _add_new_xlabel(fig,pcts,y,xticks,xlabels,label):
    new_ax = fig.add_axes((0,y,0.8,0))
    new_ax.plot(pcts,np.zeros(len(pcts)))
    new_ax.yaxis.set_visible(False)
    new_ax.spines['bottom'].set_color('gray')
    new_ax.set_xticks(xticks)
    new_ax.set_xticklabels(xlabels)
    new_ax.set_xlabel(f'{label}')
    plt.axis('tight')
    return new_ax

def _pct_to_N(pcts,num_items):
    return (pcts/100*num_items).astype(int)

def _N_to_Nk(N,k):
    return np.round((N/k),2)