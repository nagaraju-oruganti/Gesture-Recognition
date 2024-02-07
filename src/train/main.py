import os
import numpy as np
from keras.callbacks import (ModelCheckpoint,
                             ReduceLROnPlateau, 
                             EarlyStopping,
                             CSVLogger)

### Local modules
from data.dataset import Dataset
from data.generator import Generator
from models.cnn3d import ModelConv3D
from models.cnn_3d import conv3D
from utils import utils

class Trainer:
    def __init__(self, config, **kwargs):
        self.config = utils.amend_config(config = config)
        utils.seed_everything(seed = config.seed)
        self.suffix = kwargs.get('filename_suffix', '')
        if self.suffix != '': self.suuffix = f'_{self.suffix}'
        
    def prepare(self):

        # data
        train, valid = Dataset(config = self.config)()
        self.meta_data = {
            'train_size' : len(train),
            'valid_size' : len(valid),
        }
        
        # generators
        self.train_generator = Generator(config = self.config,
                                         data = train,
                                         batch_size = self.config.train_batch_size,
                                         shuffle = True).load()
        self.valid_generator = Generator(config = self.config,
                                         data = valid,
                                         batch_size = self.config.valid_batch_size,
                                         shuffle = False).load()
        
        if len(train) % self.config.train_batch_size == 0:
            self.steps_per_epoch = int(len(train) / self.config.train_batch_size)
        else:
            self.steps_per_epoch = (len(train) // self.config.train_batch_size) + 1
        
        if len(valid) % self.config.valid_batch_size == 0:
            self.validation_steps = int(len(valid) / self.config.valid_batch_size)
        else:
            self.validation_steps = (len(valid) // self.config.valid_batch_size) + 1
        
        # checkpoint caller
        checkpoint = ModelCheckpoint(
            f'{self.config.dest_path}/model{self.suffix}.keras',
            monitor = 'val_loss',
            verbose = 0,
            save_best_only = True,
            save_weights_only = False,
            mode = 'auto',
            save_freq = 'epoch'
        )
        
        # earlystop caller
        es_params = self.config.earlystopping_params
        earlystop = EarlyStopping(
            monitor = 'val_loss',
            min_delta = es_params['min_delta'],
            patience = es_params['patience'],
            verbose = 1
        )
        
        # Learning rate scheduler
        lr_params = self.config.scheduler_params
        scheduler = ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = lr_params['factor'],
            patience = lr_params['patience'],
            min_lr = lr_params['min_lr'], 
            cooldown=1, 
            verbose=1
        )
        
        # save history
        history_logger = CSVLogger(f'{self.config.dest_path}/training{self.suffix}.log',
                                   separator = ',', 
                                   append = False)

        # CALLBACKS
        self.callbacks = [earlystop, scheduler, history_logger]
        if self.config.save_checkpoint: self.callbacks.append(checkpoint)
        
    def start(self, model, use_class_weights = False):
        self.prepare()
        history = model.fit(
            self.train_generator,
            steps_per_epoch = self.steps_per_epoch,
            epochs = self.config.num_epochs,
            verbose = 1,
            validation_data = self.valid_generator,
            validation_steps = self.validation_steps,
            class_weight = self.config.class_weights if use_class_weights else None,
            workers = -1,
            initial_epoch = 0,
            callbacks = self.callbacks
        )
        return history, self.meta_data
        