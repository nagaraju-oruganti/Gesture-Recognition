import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import keras

def seed_everything(seed):
    '''apply random seed'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    
def class_weights_estimator(df):
    
    # https://naadispeaks.blog/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
    
    class_weights = []
    dist = dict(df['label'].value_counts())
    
    total_samples = len(df)
    for c in sorted(df['label'].unique()):
        class_weights.append(
            1 - (dist[c] / total_samples)
        )    
    return class_weights

def amend_config(config):
    
    # Repo to save model
    if config.models_dir != '':
        config.dest_path = os.path.join(config.models_dir, config.model_name)
        os.makedirs(config.dest_path, exist_ok=True)
    
    # Clip dimensions
    (config.frames, config.channels, config.width, config.height) = config.clip_dim
    
    return config