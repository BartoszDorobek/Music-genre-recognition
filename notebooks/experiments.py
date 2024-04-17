from datetime import datetime

import kapre
import numpy as np
import pandas as pd
import tensorflow as tf
from audiomentations import Compose
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import Dense

from data_utils import SAMPLE_RATE, DATA_PATH, EVALUATE_FILE_PATH
from experiment_config import experiments_list, get_internal_model

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

START_LR = 1e-5
MIN_LR = 1e-6


def get_data(data_type, batch_size):
    assert data_type in ['3s', '30s'], "data_type shoulbe either 3s or 30s"
    if data_type == '3s':
        audio_dir = DATA_PATH / 'genres_3_seconds'
        max_signal_length_to_crop = 67_500
    elif data_type == '30s':
        audio_dir = DATA_PATH / 'genres_original'
        max_signal_length_to_crop = 660_000
    else:
        print("wrong data type")
    input_shape = (max_signal_length_to_crop, 1)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=audio_dir,
        batch_size=batch_size,
        validation_split=0.2,
        seed=SEED,
        shuffle=True,
        output_sequence_length=max_signal_length_to_crop,
        subset='both',
        label_mode='categorical'
    )
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    return train_ds, val_ds, test_ds, input_shape


def get_spectrogram_layer(input_shape, data_type):
    if data_type == '3s':
        WIN_LENGTH = 1024 * 2
        FRAME_STEP = int(WIN_LENGTH / 4)  # / 4 a nie /2

    elif data_type == '30s':
        WIN_LENGTH = 1024 * 4
        FRAME_STEP = int(WIN_LENGTH / 2)  # / 4 a nie /2
    N_FFT = 1024 * 4  # jeszcz razy 4
    N_MELS = 128  # 128
    MEL_F_MIN = 20
    MEL_F_MAX = 7600
    RETURN_DECIBEL = True
    DATA_FORMAT = 'channels_last'

    specrtogram_layer = kapre.composed.get_melspectrogram_layer(
        input_shape=input_shape,
        win_length=WIN_LENGTH,
        hop_length=FRAME_STEP,
        n_fft=N_FFT,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        mel_f_min=MEL_F_MIN,
        mel_f_max=MEL_F_MAX,
        return_decibel=RETURN_DECIBEL,
        input_data_format=DATA_FORMAT,
        output_data_format=DATA_FORMAT
    )
    return specrtogram_layer


# def get_classification_layers(dropout, layers_sizes):
#     classifiation_layers = [
#         Flatten(),
#         sum([[
#                 Dense(units=units, activation='relu'), Dropout(rate=0.2)
#                 if dropout else Dense(units=units, activation='relu')
#             ] for units in layers_sizes],[]
#         ),
#     ]
#     return classifiation_layers


def get_model(model_type, data_type, input_shape):
    specrtogram_layer = get_spectrogram_layer(input_shape, data_type)
    model = Sequential([
        specrtogram_layer,
        *get_internal_model(model_type),
        Dense(units=10, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=START_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def get_augmentation_pipe(augmentation_type, train_ds):
    augmentations_pipeline = Compose(augmentation_type)

    def apply_pipeline(y):
        shifted = augmentations_pipeline(y, sample_rate=SAMPLE_RATE)
        return shifted

    @tf.function
    def tf_apply_pipeline(feature, label):
        """
        Applies the augmentation pipeline to audio files
        @param y: audio data
        @param sr: sampling rate
        @return: augmented audio data
        """
        feature = feature[:, :, 0]
        augmented_feature = tf.numpy_function(
            apply_pipeline, inp=[feature], Tout=tf.float32, name="apply_pipeline"
        )
        augmented_feature = tf.expand_dims(augmented_feature, axis=-1)
        return augmented_feature, label

    def augment_audio_dataset(dataset: tf.data.Dataset):
        dataset = dataset.map(tf_apply_pipeline)

        return dataset

    train_ds_aug = augment_audio_dataset(train_ds)
    return train_ds_aug


def get_callbacks(model_desc):
    earlystopping = EarlyStopping(
        monitor="val_accuracy",
        patience=20,  # increase later
        restore_best_weights=True,
        min_delta=0.008
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=8,
        min_lr=MIN_LR,
        min_delta=0.2
    )

    # factor = 0.2,
    # patience = 10,
    # min_delta = 0.1

    model_checkpoint = ModelCheckpoint(
        f"models/best_model_{model_desc}.hdf5",
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        mode='auto',
        save_freq='epoch'
    )

    csv_logger = CSVLogger(
        f'logs/history_{model_desc}.csv',
        separator=',',
        append=False
    )
    callbacks = [
        earlystopping,
        model_checkpoint,
        csv_logger,
        reduce_lr,
    ]
    return callbacks


def _get_augmentation_desc(augmentation_type):
    aug_desc = "...??? aumentation_desc_error ???..."
    if len(augmentation_type) == 0:
        aug_desc = 'no_aug'
    elif len(augmentation_type) == 1:
        aug_desc = augmentation_type[0].__class__.__name__
    elif len(augmentation_type) > 1:
        aug_desc = 'multiple_aug'
        # aug_desc = '_'.join([a.__class__.__name__ for a in augmentation_type])
    return aug_desc


def get_model_desc(exp_id, data_type, model_type, augmentation_type, batch_size):
    aug_desc = _get_augmentation_desc(augmentation_type)
    batch_size_str = f"bs_{batch_size}"
    model_desc = "__".join([exp_id, data_type, batch_size_str, model_type, aug_desc])
    return model_desc


def run_experiment(epochs, exp_id, exp_description, data_types, model_types, augmentation_types, batch_sizes):
    print(f'experiment_id: {exp_id} \nexperiment description: {exp_description}')
    for data_type in data_types:

        for batch_size in batch_sizes:
            train_ds, val_ds, test_ds, input_shape = get_data(data_type, batch_size)
            for model_type in model_types:
                for augmentation_type in augmentation_types:
                    model = get_model(model_type, input_shape=input_shape, data_type=data_type)
                    train_ds_aug = get_augmentation_pipe(augmentation_type, train_ds)
                    model_desc = get_model_desc(exp_id, data_type, model_type, augmentation_type, batch_size)
                    print(f'\nModel Training - {model_desc}')
                    start_time = datetime.now()

                    history = model.fit(
                        train_ds_aug,  # train_ds, train_ds_aug
                        epochs=epochs,
                        validation_data=val_ds,
                        callbacks=get_callbacks(model_desc)
                    )
                    end_time = datetime.now()
                    training_time = end_time - start_time
                    n_epochs = len(history.history['val_loss'])
                    print('\nTest Dataset Evaluation')
                    test_loss, test_acc = model.evaluate(test_ds)
                    print('\nTrain Dataset Evaluation')
                    train_loss, train_acc = model.evaluate(train_ds_aug)
                    print('\nVal Dataset Evaluation')
                    val_loss, val_acc = model.evaluate(val_ds)
                    evaluate_dict = {
                        'exp_id': exp_id,
                        'model_desc': model_desc,
                        'start_time': start_time.strftime('%Y_%m_%d_%H_%M_%S'),
                        'end_time': end_time.strftime('%Y_%m_%d_%H_%M_%S'),
                        'training_length': str(training_time).split('.')[0],
                        'batch_size': batch_size,
                        'data_type': data_type,
                        'model_type': model_type,
                        'n_epochs': n_epochs,
                        'test_acc': test_acc,
                        'test_loss': test_loss,
                        'train_acc': train_acc,
                        'train_loss': train_loss,
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                    }
                    # if file does not exist than create and add header
                    if not EVALUATE_FILE_PATH.is_file():
                        header = pd.DataFrame(columns=[
                            'exp_id', 'model_desc', 'start_time', 'end_time', 'training_length',
                            'batch_size', 'data_type', 'model_type', 'n_epochs',
                            'test_acc', 'test_loss',
                            'train_acc', 'train_loss',
                            'val_acc', 'val_loss',
                        ])
                        header.to_csv(EVALUATE_FILE_PATH, mode='a', index=False, header=True)
                    df = pd.DataFrame(evaluate_dict, index=[0])
                    df.to_csv(EVALUATE_FILE_PATH, mode='a', index=False, header=False)


def run_experiments(epochs):
    for experiment in experiments_list:
        if experiment['exp_id'] in ['1_2']:
            run_experiment(epochs=epochs, **experiment)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    run_experiments(epochs=120)
