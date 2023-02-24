import os
import numpy as np
import tensorflow as tf

from config import ModelConfig

from glob import glob
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, add
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import L2

# mixed precision
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')
# mixed_precision

GPU_LIMIT = 1024*4  # MB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:  # 텐서플로가 첫 번째 GPU에 GPU_LIMIT 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_LIMIT)]
        )
    except RuntimeError as e:
        print(e)

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = np.random.permutation(self.x.shape[0])
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        indices = self.idx[start_idx:end_idx]
        
        batch_x = self.x[indices]
        batch_y = self.y[indices]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def tuner_model(hp) -> keras.models.Model:
    """
    :return: built keras model
    """
    inputs = Input(shape=(ModelConfig.n_day, ModelConfig.n_feature), name='finance_data', dtype='float32')
    blk_output = None

    conv_res_hp_blk = hp.Int('conv_blocks', min_value=2, max_value=16, step=2)
    conv_hp_kernel = hp.Choice('filters', values=[16, 32, 64, 128, 256])
    kernel_size_hp = hp.Choice('kernel_size', values=[3, 5, 7, 11])
    
    dense_hp_blk = hp.Int('dense_blocks', min_value=1, max_value=4, step=1)
    dense_hp_units = hp.Choice('units', values=[32, 64, 128, 256, 512])
    dropout_hp_rate = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3, 0.4, 0.5])
    l2_hp_val = hp.Choice('l2_val', values=['None', '1e-2', '1e-4', '1e-6']) 
    
    if l2_hp_val == 'None':
        kernel_regularizer = None
        bias_regularizer = None
    elif l2_hp_val == '1e-2':
        kernel_regularizer = L2(1e-2)
        bias_regularizer = L2(1e-2)
    elif l2_hp_val == '1e-4':
        kernel_regularizer = L2(1e-4)
        bias_regularizer = L2(1e-4)
    elif l2_hp_val == '1e-6':
        kernel_regularizer = L2(1e-6)
        bias_regularizer = L2(1e-6)
    
    layer_config = { **ModelConfig.layers_config }
    layer_config['kernel_regularizer'] = kernel_regularizer
    layer_config['bias_regularizer'] = bias_regularizer
    conv_config = { **ModelConfig.conv_layer_config, **layer_config }
    conv_config['filters'] = conv_hp_kernel
    conv_config['kernel_size'] = kernel_size_hp
    dropout_config = { **ModelConfig.dropout_config }
    dropout_config['rate'] = dropout_hp_rate
        
    for i in range(conv_res_hp_blk):
        if i == 0:
            x = ModelConfig.layers['conv'](**conv_config)(inputs)
            x = ModelConfig.layers['batchnorm']()(x)
            x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
            x = ModelConfig.layers['conv'](**conv_config)(x)
            x = ModelConfig.layers['batchnorm']()(x)
            blk_output = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
        else:
            x = ModelConfig.layers['conv'](**conv_config)(blk_output)
            x = ModelConfig.layers['batchnorm']()(x)
            x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
            x = ModelConfig.layers['conv'](**conv_config)(x)
            x = ModelConfig.layers['batchnorm']()(x)
            x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
            blk_output = add([x, blk_output], name=f'cnn_res_block_{i}')

    x = ModelConfig.layers['pooling'](**ModelConfig.pooling_config)(blk_output)

    for i in range(dense_hp_blk):
        x = ModelConfig.layers['dense'](dense_hp_units, **layer_config)(x)
        x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
        x = ModelConfig.layers['dropout'](**dropout_config)(x)

    outputs = Dense(ModelConfig.n_future, activation='linear', name='predicted_price', dtype='float32')(x)

    model = models.Model(inputs, outputs, name='finance')
    
    hp_lr = hp.Choice('learning_rate', values=[5e-3, 1e-3, 5e-4, 1e-4]) 
    optimizer = keras.optimizers.Adam(learning_rate=hp_lr)
    
    model.compile(
        optimizer=optimizer,
        loss=ModelConfig.loss,
        metrics=ModelConfig.metrics
    )
    
    return model

    
def construct_model(input_shape: tuple) -> keras.models.Model:
    """
    :param input_shape: shape of input data
    :return: built keras model
    """
    inputs = Input(shape=input_shape, name='finance_data', dtype='float32')
    blk_output = None

    for i in range(ModelConfig.num_conv_res_block):
        if i == 0:
            x = ModelConfig.layers['conv'](**ModelConfig.conv_layer_config, **ModelConfig.layers_config)(inputs)
            x = ModelConfig.layers['batchnorm']()(x)
            x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
            x = ModelConfig.layers['conv'](**ModelConfig.conv_layer_config, **ModelConfig.layers_config)(x)
            x = ModelConfig.layers['batchnorm']()(x)
            blk_output = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
        else:
            x = ModelConfig.layers['conv'](**ModelConfig.conv_layer_config, **ModelConfig.layers_config)(blk_output)
            x = ModelConfig.layers['batchnorm']()(x)
            x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
            x = ModelConfig.layers['conv'](**ModelConfig.conv_layer_config, **ModelConfig.layers_config)(x)
            x = ModelConfig.layers['batchnorm']()(x)
            x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
            blk_output = add([x, blk_output], name=f'cnn_res_block_{i}')

    x = ModelConfig.layers['pooling'](**ModelConfig.pooling_config)(blk_output)

    for i in range(ModelConfig.num_dense_layer):
        x = ModelConfig.layers['dense'](ModelConfig.NUM_DENSE_WIDTH, **ModelConfig.layers_config)(x)
        x = ModelConfig.layers['activation'](**ModelConfig.activation_config)(x)
        x = ModelConfig.layers['dropout'](**ModelConfig.dropout_config)(x)

    outputs = Dense(
        ModelConfig.n_future, activation='linear',
        name='predicted_price', dtype='float32'
    )(x)

    model = models.Model(inputs, outputs, name='finance')
    
    model.compile(
        optimizer=ModelConfig.optimizer,
        loss=ModelConfig.loss,
        metrics=ModelConfig.metrics
    )    
    return model


def get_last_part_of_path(path):
    return os.path.basename(os.path.normpath(path))


def find_best_model(model_names: list, order_key=None):
    """
    :param model_names: list. glob of saved model dirs
    :param order_key: str. by the order_key, find best model.
    :return: str. best model name found by the given order_key in model_names.
    """

    if order_key is None:
        order_key = ModelConfig.monitor

    best_val_loss = np.inf
    best_index = -1

    for idx, model_name in enumerate(model_names):
        results = get_last_part_of_path(
            model_name.replace(ModelConfig.model_name, '').replace(ModelConfig.saved_model_ext, '')
        )[1:]
        splitted_results = results.split(ModelConfig.model_result_separator)
        for result in splitted_results:
            key, value = result.split(ModelConfig.model_name_separator)
            if key != order_key:
                continue

            value = float(value)
            if value < best_val_loss:
                best_val_loss = value
                best_index = idx
    if best_index != -1:
        print(f'found best model: {model_names[best_index]}')
        return model_names[best_index]
    else:
        print('model not found')
        return None


def get_saved_model_names() -> list:
    saved_models = glob(
        os.path.join(
            ModelConfig.model_path, '*' + ModelConfig.saved_model_ext
        )
    )
    return saved_models


def get_model(input_shape: tuple, load='best') -> keras.Sequential:
    """
    :param input_shape: tuple. (time_steps, number of features of data)
    :param load: str. only best available. load model option.
    :return: sequential model in keras
    """
    saved_models = get_saved_model_names()
    model = None

    if saved_models:
        if load == 'best':
            model_name = find_best_model(saved_models)
            model = models.load_model(
                model_name, custom_objects=ModelConfig.custom_objects
            )
    else:
        print('model not found.')
        model = construct_model(input_shape)
    return model


def remove_models(exception='best'):
    saved_models = get_saved_model_names()

    survivor = ''
    if saved_models:
        if exception == 'best':
            survivor = find_best_model(saved_models)
        saved_models.pop(saved_models.index(survivor))

        for victim_model in saved_models:
            print(victim_model, 'removed.')
            os.remove(victim_model)








