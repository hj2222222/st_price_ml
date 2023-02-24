import os
import pickle

import numpy as np
import pandas as pd
from tqdm import notebook


def dataloader(_dirs, prefix, postfixs=None):
    ext = '.pickle'
    if postfixs is None:
        postfixs = ['x_train', 'x_test', 'y_train', 'y_test']
    dataset = []
    for postfix in postfixs:
        data_path = os.path.join(_dirs, f'{prefix}_{postfix}{ext}')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        dataset.append(data)
    return dataset


def get_data(n_day, n_future, data_path, drop=None, fill='bfill', output_column='Close'):
    df = pd.read_csv(data_path).fillna(method=fill)
    if drop is not None:
        df.drop(drop, axis=1, inplace=True)
    output_idx = df.columns.get_loc(output_column)
    length_data = len(df)
    n_data = max(length_data - n_day - n_future, 0)
    n_feature = len(df.columns)
    
    # allocation prevents resizing
    x_data = np.zeros((n_data, n_day, n_feature), np.float32)
    y_data = np.zeros((n_data, n_future), np.float32)

    # create time-series dataset
    prog_bar = notebook.tqdm(range(n_data), desc='data loading', leave=False)
    for day in prog_bar:
        x = df.iloc[day:day+n_day]
        y = df.iloc[day+n_day:day+n_day+n_future, output_idx]
        x_data[day, :] = x
        y_data[day, :] = y

    return x_data, y_data


if __name__ == '__main__':
    pass
    # get_data(32)
