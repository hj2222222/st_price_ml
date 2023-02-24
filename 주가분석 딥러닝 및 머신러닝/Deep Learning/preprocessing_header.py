import ta
from numpy import log10

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, FunctionTransformer


def log_transform(data):
    return log10(data)


def div100_transform(data):
    return data / 100.


SCALER_EXT = '.sclr'
SCALER_DIR = './revised_dataset/scaler/'

DATA_EXT = '.csv'
DATA_DIR = './revised_dataset/original_csv/'

SCALED_DATA_DIR = './revised_dataset/scaled_csv/'

COLUMNS = [
    'Date', 'Open', 'High', 'Low',
    'Close', 'Volume', 'Change',
    'C-Close', 'C-Change', 'I-Close', 'I-Change'
]

INDICATORS = [
    'MFI', 'ADI', 'OBV', 'MACD',
    'CCI', 'RSI', 'BOL_H', 'BOL_L',
    'ma_5', 'ma_20', 'ma_60', 'ma_120'
]

COLUMN_SCALER = {
    'Open': MinMaxScaler(),
    'High': MinMaxScaler(),
    'Low': MinMaxScaler(),
    'Close': MinMaxScaler(),
    'Volume': MinMaxScaler(),
    'Change': FunctionTransformer(div100_transform),  # percent unit
    'C-Close': 'default',  # minMax
    'C-Change': FunctionTransformer(div100_transform),  # percent unit
    'I-Close': 'default',  # minMax
    'I-Change': FunctionTransformer(div100_transform),  # percent unit
    # indicators
    'MFI': FunctionTransformer(div100_transform),  # range 0 ~ 100
    'ADI': MaxAbsScaler(),
    'OBV': MaxAbsScaler(),
    'MACD': MaxAbsScaler(),
    'CCI': MaxAbsScaler(),
    'RSI': FunctionTransformer(div100_transform),  # range 0 ~ 100
    'BOL_H': MinMaxScaler(),
    'BOL_L': MinMaxScaler(),
    'ma_5': MinMaxScaler(),
    'ma_20': MinMaxScaler(),
    'ma_60': MinMaxScaler(),
    'ma_120': MinMaxScaler(),
}

INDICATOR_TO_FUNCTION = {
    'MFI': (ta.volume.money_flow_index, ('high', 'low', 'close', 'volume')), 
    'ADI': (ta.volume.acc_dist_index, ('high', 'low', 'close', 'volume')),
    'OBV': (ta.volume.on_balance_volume, ('close', 'volume')),
    'CMF': (ta.volume.chaikin_money_flow, ('high', 'low', 'close', 'volume')),
    'FI' : (ta.volume.force_index, ('close', 'volume')),
    'EOM': (ta.volume.ease_of_movement, ('high', 'low', 'volume')),
    'VPT': (ta.volume.volume_price_trend, ('close', 'volume')),
    'NVI': (ta.volume.negative_volume_index, ('close', 'volume')),
    'VMAP': (ta.volume.volume_weighted_average_price, ('high', 'low', 'close', 'volume')),
    # Volatility
    'ATR': (ta.volatility.average_true_range, ('high', 'low', 'close')),
    'BHB': (ta.volatility.bollinger_hband, ('close', )),
    'BLB': (ta.volatility.bollinger_lband, ('close', )),
    'KCH': (ta.volatility.keltner_channel_hband, ('high', 'low', 'close')),
    'KCL': (ta.volatility.keltner_channel_lband, ('high', 'low', 'close')),
    'KCM': (ta.volatility.keltner_channel_mband, ('high', 'low', 'close')),
    'DCH': (ta.volatility.donchian_channel_hband, ('high', 'low', 'close')),
    'DCL': (ta.volatility.donchian_channel_lband, ('high', 'low', 'close')),
    'DCM': (ta.volatility.donchian_channel_mband, ('high', 'low', 'close')),
    'UI' : (ta.volatility.ulcer_index, ('close', )),
    # Trend
    'SMA': (ta.trend.sma_indicator, ('close', )),
    'EMA': (ta.trend.ema_indicator, ('close', )),
    'WMA': (ta.trend.wma_indicator, ('close', )),
    'MACD': (ta.trend.macd, ('close', )),
    'ADX': (ta.trend.adx, ('high', 'low', 'close')),
    '-VI': (ta.trend.vortex_indicator_neg, ('high', 'low', 'close')),
    '+VI': (ta.trend.vortex_indicator_pos, ('high', 'low', 'close')),
    'TRIX': (ta.trend.trix, ('close', )),
    'MI': (ta.trend.mass_index, ('high', 'low')),
    'CCI': (ta.trend.cci, ('high', 'low', 'close')),
    'DPO': (ta.trend.dpo, ('close', )),
    'KST': (ta.trend.kst, ('close', )),
    'Ichimoku': (ta.trend.ichimoku_a, ('high', 'low')),
    'Parabolic SAR': (ta.trend.psar_down, ('high', 'low', 'close')),
    'STC': (ta.trend.stc, ('close', )),
    # Momentum
    'RSI': (ta.momentum.rsi, ('close', )),
    'SRSI': (ta.momentum.stochrsi, ('close', )),
    'TSI': (ta.momentum.tsi, ('close', )),
    'UO': (ta.momentum.ultimate_oscillator, ('high', 'low', 'close')),
    'SR': (ta.momentum.stoch, ('high', 'low', 'close')),
    'WR': (ta.momentum.williams_r, ('high', 'low', 'close')),
    'AO': (ta.momentum.awesome_oscillator, ('high', 'low')),
    'KAMA': (ta.momentum.kama, ('close', )),
    'ROC': (ta.momentum.roc, ('close', )),
    'PPO': (ta.momentum.ppo, ('close', )),
    'PVO': (ta.momentum.pvo, ('volume', )),
    'BOL_H': (ta.volatility.bollinger_hband, ('close', )),
    'BOL_L': (ta.volatility.bollinger_lband, ('close', )),
}

