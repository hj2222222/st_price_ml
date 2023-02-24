import os
import IPython

from datetime import datetime

from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LeakyReLU, Dropout, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard, Callback


class ClearTrainingOutput(Callback):
    def on_test_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


class ModelConfig:
    # general
    seed = 42
    epochs = 100
    ES_PATIENCE = 150  # early stopping patience
    RP_PATIENCE = 100  # reduce LR on plateau patience
    batch_size = 4096
    val_split = 0.2
    
    # data shapes
    # x_data.shape == (None, n_day, n_feature)
    # y_data.shape == (None, n_future)
    n_day = 32
    n_feature = 22
    n_future = 32

    # model settings
    # save and load
    model_name = 'AcademyProject'
    log_path = './logs/'
    model_path = './models/'
    saved_model_ext = '.hdf5'
    saved_model_metrics = {
        'epoch': '{epoch:03d}',
        'val_loss': '{val_loss:.10f}',
    }
    model_name_separator = '-'
    model_result_separator = '+'

    saved_model_name = os.path.join(
        model_path, model_name_separator.join(
            [
                model_name,
                model_result_separator.join(
                    [
                        f'{metric}-{metric_format}'
                        for metric, metric_format in saved_model_metrics.items()
                    ]
                )
            ]
        ) + saved_model_ext
    )

    # compile
    learning_rate = 5e-4
    reduced_learning_rate_factor = 0.9
    min_learning_rate = 1e-6
    optimizer = Adam(learning_rate=learning_rate)  #, clipnorm=1.)
    loss = 'mse'
    metrics = ['mae']

    # callbacks
    monitor = 'val_loss'
    callback_mode = 'min'

    model_checkpoint = ModelCheckpoint(
        saved_model_name,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode=callback_mode
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=ES_PATIENCE,
        mode=callback_mode,
        restore_best_weights=False,
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor=monitor,
        factor=reduced_learning_rate_factor,
        patience=RP_PATIENCE,
        verbose=1,
        mode=callback_mode,
        min_lr=min_learning_rate,
    )
    terminate_nan = TerminateOnNaN()

    tensorboard_logdir = os.path.join(log_path, model_name)
    now_date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    tensorboard_callback = TensorBoard(
        log_dir='_'.join([tensorboard_logdir, now_date])
    )

    output_clear = ClearTrainingOutput()
    
    callbacks = [
        output_clear, model_checkpoint, early_stopping,
        reduce_on_plateau, terminate_nan, tensorboard_callback,
    ]

    # layers
    # Conv1D
    NUM_FILTERS = 128
    KERNEL_SIZE = 3
    strides = 1
    padding = 'same'

    # MaxPooling
    pool_size = 2
    pooling_config = {
        # 'pool_size': pool_size,
    }

    # Dense layers
    dropout_rate = 0.1
    NUM_DENSE_WIDTH = 256
    dropout_config = {
        'rate': dropout_rate,
        'seed': seed,
    }

    # activations
    alpha = 0.1
    activation = LeakyReLU
    activation_config = {
        'alpha': alpha,
    }

    # layers
    num_conv_res_block = 4
    num_dense_layer = 1

    layers_config = {
        'kernel_initializer': initializers.he_uniform(seed=seed),
#         'kernel_initializer': initializers.RandomNormal(seed=seed),
        'bias_initializer': initializers.Zeros(),
        'kernel_regularizer': L2(1e-6),
        'bias_regularizer': L2(1e-6),
    }
    conv_layer_config = {
        'filters': NUM_FILTERS,
        'kernel_size': KERNEL_SIZE,
        'strides': strides,
        'padding': padding,
    }

    layers = {
        'conv': Conv1D,
        'dense': Dense,
        'activation': activation,
        'dropout': Dropout,
        'pooling': GlobalAveragePooling1D,
        'batchnorm': BatchNormalization,
    }

    # model custom objects
    custom_objects = {
        'LeakyReLU': LeakyReLU,
        'loss': Huber
    }





















