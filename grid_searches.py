from ray import tune

from dl_generators import *

feature_spaces = {'distances': [features_distances, [91]], 
                    'identity': [features_identity, [2,7,2]],
                    'distances_normalized': [features_distances_normalized, [91]],
                    'mars': [features_mars, []],
                    'mars_distr': [features_mars_distr, []],
                    'mars_no_shift': [features_mars_no_shift, [160]]
                }

sweeps_baseline = {
    'test_run':
        [{'num_samples': 1},
        {
            "model_param__learning_rate": tune.choice([0.001, 0.0005]),
            "epochs": 3
        }],
    'test_run_distances':
        [{'num_samples': 1},
        {
            "model_param__learning_rate": tune.choice([0.0001]),
            "epochs": 15,
            "features": "distances",
            "model_param__dropout_rate": 0.5
        }],
    'test_run_distances_normalized':
        [{'num_samples': 1},
        {
            "model_param__learning_rate": tune.choice([0.0001]),
            "epochs": 15,
            "features": "distances_normalized",
            "model_param__dropout_rate": 0.5
        }],
    'full_run_distance_deeper_moredropout':
        [{'num_samples': 200},
        {
            "future_frames": tune.randint(40, 70),
            "past_frames": tune.randint(40, 70),
            "model_param__learning_rate": tune.loguniform(0.0001, 0.001),
            "model_param__conv_size": tune.randint(3, 7),
            "model_param__dropout_rate": tune.uniform(0.2, 0.8),
            "model_param__layer_channels": tune.choice([(128,64, 32), (128,64,64),(256,128,64),(256,128,128)]),
            "reweight_loss": tune.choice([True, False]),
            "epochs": 30,
            "features": "distances",
            "learning_decay_freq": 5
        }
        ],
    'full_run':
        [{'num_samples': 1000},
        {
            "future_frames": tune.randint(50, 150),
            "past_frames": tune.randint(50, 150),
            "model_param__learning_rate": tune.loguniform(0.0001, 0.001),
            "model_param__conv_size": tune.randint(3, 16),
            "model_param__layer_channels": tune.choice([(512,256), (512,64),(1024,256),(128,64)]),
            "reweight_loss": tune.choice([True, False]),
            "epochs": 50
        }
        ],
    'full_run_deeper':
        [{'num_samples': 200},
        {
            "future_frames": tune.randint(40, 100),
            "past_frames": tune.randint(40, 100),
            "model_param__learning_rate": tune.loguniform(0.0001, 0.001),
            "model_param__conv_size": tune.randint(3, 16),
            "model_param__dropout_rate": tune.uniform(0.2, 0.5),
            "model_param__layer_channels": tune.choice([(128,64, 32), (128,64,64),(256,128,64),(256,128,128)]),
            "reweight_loss": tune.choice([True, False]),
            "epochs": 50,
            "learning_decay_freq": 10
        }
        ],
    'full_run_deeper_moredropout':
        [{'num_samples': 200},
        {
            "future_frames": tune.randint(40, 100),
            "past_frames": tune.randint(40, 100),
            "model_param__learning_rate": tune.loguniform(0.0001, 0.001),
            "model_param__conv_size": tune.randint(3, 16),
            "model_param__dropout_rate": tune.uniform(0.2, 0.8),
            "model_param__layer_channels": tune.choice([(128,64, 32), (128,64,64),(256,128,64),(256,128,128)]),
            "reweight_loss": tune.choice([True, False]),
            "epochs": 50,
            "learning_decay_freq": 10
        }
        ],
    'full_run_distances_normalized':
        [{'num_samples': 200},
        {
            "future_frames": tune.randint(40, 100),
            "past_frames": tune.randint(40, 100),
            "model_param__learning_rate": tune.loguniform(0.0001, 0.001),
            "model_param__conv_size": tune.randint(3, 16),
            "model_param__dropout_rate": tune.uniform(0.2, 0.8),
            "model_param__layer_channels": tune.choice([(128,64, 32), (128,64,64),(256,128,64),(256,128,128)]),
            "reweight_loss": tune.choice([True, False]),
            "epochs": 30,
            "learning_decay_freq": 10,
            "batch_size": 64,
            "features": "distances_normalized"
        }],
    'test_run_mars_no_shift':
        [{'num_samples': 1},
        {
            "model_param__learning_rate": tune.choice([0.00025]),
            "epochs": 30,
            "features": "mars_no_shift",
            "future_frames": 30,
            "past_frames": 30,
            "model_param__dropout_rate": 0.8,
            "normalize": False
        }],
}