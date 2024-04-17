from audiomentations import AddGaussianNoise, Shift, BandStopFilter, ClippingDistortion
from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

gaussian = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
shift = Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
band_stop = BandStopFilter()
distortion = ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=20, p=0.2)


def get_internal_model(model_type):
    internal_layers = []
    if model_type == 'CNN_Basic':
        internal_layers = [
            Conv2D(filters=8, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_Average':
        internal_layers = [
            Conv2D(filters=8, kernel_size=3, activation='relu'),
            AveragePooling2D(2),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            AveragePooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            AveragePooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_Dropout':
        internal_layers = [
            Conv2D(filters=8, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Dropout(0.2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Dropout(0.2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(0.2),
        ]
    # ---------------------------------------------------------
    elif model_type == 'CNN_Basic_D_128_80':
        internal_layers = [
            Conv2D(filters=8, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    elif model_type == 'CNN_Basic_D_256':
        internal_layers = [
            Conv2D(filters=8, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Basic_D_256_80':
        internal_layers = [
            Conv2D(filters=8, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    # -------------------------------------------------------
    elif model_type == 'CNN_32_64_D_128':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_D_128_80':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_D_256':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_D_256_80':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    # -------------------------------------------------------
    elif model_type == 'CNN_16_32_64_D_128':
        internal_layers = [
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_16_32_64_D_128_80':
        internal_layers = [
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    elif model_type == 'CNN_16_32_64_D_256':
        internal_layers = [
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_16_32_64_D_256_80':
        internal_layers = [
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    # ---------------------------------------------------------
    elif model_type == 'CNN_32_64_128_D_128':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_128_D_128_80':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_128_D_256':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_128_D_256_80':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    # ---------------------------------------------------------
    elif model_type == 'CNN_64_64_64_D_128':
        internal_layers = [
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_64_64_64_D_128_80':
        internal_layers = [
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    elif model_type == 'CNN_64_64_64_D_256':
        internal_layers = [
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_64_64_64_D_256_80':
        internal_layers = [
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=80, activation='relu'),
        ]
    # ---------------------------------------------------------
    elif model_type == 'CNN_Best_K_5_3':
        internal_layers = [
            Conv2D(filters=32, kernel_size=5, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_5_5':
        internal_layers = [
            Conv2D(filters=32, kernel_size=5, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=5, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_7_3':
        internal_layers = [
            Conv2D(filters=32, kernel_size=7, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_7_5':
        internal_layers = [
            Conv2D(filters=32, kernel_size=7, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=5, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_7_7':
        internal_layers = [
            Conv2D(filters=32, kernel_size=7, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=7, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    # --------------------------------------------------------
    elif model_type == 'CNN_32_64_D_256_128':
        internal_layers = [
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_32_64_64_D_256_d02_128_K_5_3_3':
        internal_layers = [
            Conv2D(filters=32, kernel_size=5, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dropout(0.2),
            Dense(units=128, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_9_9':
        internal_layers = [
            Conv2D(filters=32, kernel_size=9, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=9, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_11_11':
        internal_layers = [
            Conv2D(filters=32, kernel_size=11, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=11, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    elif model_type == 'CNN_Best_K_15_15':
        internal_layers = [
            Conv2D(filters=32, kernel_size=15, activation='relu'),
            MaxPooling2D(2),
            Conv2D(filters=64, kernel_size=15, activation='relu'),
            MaxPooling2D(2),
            Flatten(),
            Dense(units=256, activation='relu'),
        ]
    # ---------------------------------------------------------
    return internal_layers


experiments_list = [
    {
        "exp_id": '1_0',
        "exp_description": 'porównanie metod augmentacji na bazowej architekturze',
        "data_types": ['30s', '3s'],
        "batch_sizes": [32],
        "model_types": ['CNN_Basic'],
        "augmentation_types": [
            [],
            [gaussian],
            [shift],
            [band_stop],
            [distortion],
            [gaussian, shift, band_stop, distortion]
        ],
    },
    {
        "exp_id": '2_0',
        "exp_description": 'porównanie batch size na bazowej architekturze',
        "data_types": ['30s', '3s'],
        "batch_sizes": [16, 32, 64, 128],
        "model_types": ['CNN_Basic'],
        "augmentation_types": [
            [shift, band_stop, distortion]
        ],
    },

    {
        "exp_id": '3_0',
        "exp_description": 'podsatwowe porównanie modeli z pełną augmentacji ',
        "data_types": ['3s', '30s'],
        "batch_sizes": [16],
        "model_types": [
            'CNN_Basic',
            'CNN_Average',
            'CNN_Basic_D_128_80',
            'CNN_Basic_D_256',
            'CNN_Basic_D_256_80',
            'CNN_32_64_D_128',
            'CNN_32_64_D_128_80',
            'CNN_32_64_D_256',
            'CNN_32_64_D_256_80',
            'CNN_16_32_64_D_128',
            'CNN_16_32_64_D_128_80',
            'CNN_16_32_64_D_256',
            'CNN_16_32_64_D_256_80',
            'CNN_32_64_128_D_128',
            'CNN_32_64_128_D_128_80',
            'CNN_32_64_128_D_256',
            'CNN_32_64_128_D_256_80',
            'CNN_64_64_64_D_128',
            'CNN_64_64_64_D_128_80',
            'CNN_64_64_64_D_256',
            'CNN_64_64_64_D_256_80'
        ],
        "augmentation_types": [
            [shift, band_stop, distortion],
        ],
    },
    {
        "exp_id": '4_0',
        "exp_description": 'best model porównanie romiaru filtrów',
        "data_types": ['3s'],
        "batch_sizes": [16],
        "model_types": [
            'CNN_Best_K_5_3',
            'CNN_Best_K_5_5',
        ],
        "augmentation_types": [
            [shift, band_stop, distortion],
        ],
    },
    {
        "exp_id": '5_0',
        "exp_description": 'dodatkowe testy na wieksze architektury',
        "data_types": ['3s'],
        "batch_sizes": [16],
        "model_types": [
            'CNN_32_64_D_256_128',
            'CNN_32_64_64_D_256_d02_128_K_5_3_3',
        ],
        "augmentation_types": [
            [shift, band_stop, distortion],
        ],
    },
    {
        "exp_id": '4_1',
        "exp_description": 'best model porównanie romiaru filtrów',
        "data_types": ['3s'],
        "batch_sizes": [16],
        "model_types": [
            'CNN_Best_K_7_3',
            'CNN_Best_K_7_5',
            'CNN_Best_K_7_7',
        ],
        "augmentation_types": [
            [shift, band_stop, distortion],
        ],
    },
    {
        "exp_id": '4_2',
        "exp_description": 'best model porównanie romiaru filtrów',
        "data_types": ['3s'],
        "batch_sizes": [16],
        "model_types": [
            'CNN_Best_K_9_9',
            # 'CNN_Best_K_9_7',
            'CNN_Best_K_11_11',
            # 'CNN_Best_K_11_9',
            'CNN_Best_K_15_15',
            # 'CNN_Best_K_15_11',
        ],
        "augmentation_types": [
            [shift, band_stop, distortion],
        ],
    },
    {
        "exp_id": '1_2',
        "exp_description": 'porównanie metod augmentacji na bazowej architekturze',
        "data_types": ['3s'],
        "batch_sizes": [16],
        "model_types": ['CNN_Basic'],
        "augmentation_types": [
            [],
            [shift, band_stop, distortion]
        ],
    },
]
