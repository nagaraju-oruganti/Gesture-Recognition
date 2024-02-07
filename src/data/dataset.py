import os
import pandas as pd
import numpy as np
import random
import glob
import math

# Local modules
from utils import utils

class Dataset:
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        
    def load_data(self):
        train = pd.read_csv(os.path.join(self.config.data_dir, 'processed', 'train.csv'))
        valid = pd.read_csv(os.path.join(self.config.data_dir, 'processed', 'val.csv'))
        
        # Ablation study
        train, valid = self.ablation(train = train, valid = valid, seed = self.seed, size = self.config.ablation_size)
        return train, valid
    
    @staticmethod
    def ablation(train, valid, seed, size : tuple):
        # Create samples while ensuring labels are uniform
        # Sample size : None means all samples are considered
        train_n, valid_n = size
        if train_n is not None:
            train = pd.concat([train[train['label'] == l].sample(frac = 1, random_state = seed).head(train_n)\
                                for l in train['label'].unique()], axis = 0)
        if valid_n is not None:
            valid = pd.concat([valid[valid['label'] == l].sample(frac = 1, random_state = seed).head(valid_n)\
                               for l in valid['label'].unique()], axis = 0)
        return train, valid
    
    def prepare_data(self):
        train, valid = self.load_data()
        train['augment'] = False
        valid['augment'] = False
        aug_size = self.config.aug_size
        if aug_size != 0:
            '''Augmentation only happens when trained with full data
            '''
            aug_size = (math.ceil(aug_size * len(train))) if isinstance(aug_size, float) else aug_size
            aug_df = train.copy().sample(aug_size, random_state = self.seed)
            aug_df['augment'] = True
            train = pd.concat([train, aug_df], axis = 0)
            train = train.sample(frac = 1, random_state = self.seed)
            train.reset_index(drop = True, inplace = True)

        return train, valid
        
    def __call__(self):
        return self.prepare_data()