import pickle
import os

from copy import deepcopy
import pandas as pd

from preprocessing_header import *


def scale_data(df, idx, scaler_save=True) -> pd.DataFrame:
    scaled_data = pd.DataFrame(index=df.index)
    if scaler_save:
        save_scalers = dict()

    is_loaded_scaler = True
    scalers = load_scaler(idx, 'scalers')
    if scalers is None:
        scalers = deepcopy(COLUMN_SCALER)
        is_loaded_scaler = False

    for column in df.columns:
        scaler = scalers[column]

        reshaped_data = df[column].values.reshape(-1, 1)
        if reshaped_data.shape[0] == 0:
            # print(f'{idx} passed because of no data')
            return None

        if scaler is None:
            scaled_data[column] = df[column]
            
        elif scaler == 'default':
            scaler = get_default_scaler(column)
            scaled_data[column] = scaler.transform(reshaped_data)

        else:
            if not is_loaded_scaler:
                scaler.fit(reshaped_data)
            scaled_data[column] = scaler.transform(reshaped_data)
        save_scalers[column] = scaler
    
    if not is_loaded_scaler and scaler_save:
        save_scaler(idx, save_scalers, scaler_name='scalers')

    return scaled_data

def get_indicator_ma(close, n_days):
    ma = pd.DataFrame()
    ma[f'ma_{n_days}'] = close.rolling(window=n_days).mean()
    return ma


def get_indicator(name: str, fillna=True, **kwargs) -> pd.DataFrame:
    if name.startswith('ma_'):
        n_days = int(name.split('_')[1])
        return get_indicator_ma(kwargs['close'], n_days)
    else:
        func_kwargs = {}
        func, args = INDICATOR_TO_FUNCTION[name]
        for arg in args:
            func_kwargs[arg] = kwargs[arg]
        return func(fillna=fillna, **func_kwargs)


def get_default_scaler(scaler_name):
    scaler_path = get_scaler_path('', scaler_name)
    if not os.path.exists(scaler_path):
        raise ValueError(f'Default scaler {scaler_name} dose not exist.\n{scaler_path} must exists.')

    return load_scaler('', scaler_name)


def get_scaler_path(idx, scaler_name):
    if idx:
        return os.path.join(SCALER_DIR, f'{idx}_{scaler_name + SCALER_EXT}')
    else:
        return os.path.join(SCALER_DIR, f'{scaler_name + SCALER_EXT}')


def load_scaler(idx, scaler_name):
    scaler = None
    scaler_path = get_scaler_path(idx, scaler_name)

    if not os.path.exists(scaler_path):
        return scaler
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return scaler


def save_scaler(idx, scaler, scaler_name):
    scaler_path = get_scaler_path(idx, scaler_name)
    # if not os.path.exists(os.path.join(SCALER_DIR, idx)):
    #     os.mkdir(os.path.join(SCALER_DIR, idx))

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


def load_csv_data(data_path):
    df = pd.read_csv(data_path, index_col='Date', usecols=COLUMNS)

    return df



