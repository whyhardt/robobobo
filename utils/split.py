import os.path

import pandas as pd

'''
1. load a dataset with sequential data
2. split it into chunks of length n
3. save each chunk as a new dataset with the name containing the index of the first row
'''

if __name__ == '__main__':
    # load the dataset
    path = '../stock_data'
    file = 'stocks_sp500_2010_2020.csv'
    df = pd.read_csv(os.path.join(path, file))

    # split the dataset into chunks of length n
    n = 100
    df_list = [df[i:i+n] for i in range(0, df.shape[0], n)]

    # save each chunk as a new dataset with the name containing the index of the first row
    path_save = '../stock_data/stock_chunks'
    for i, df in enumerate(df_list):
        df.to_csv(os.path.join(path_save, 'data_{}.csv'.format(i)), index=False)
