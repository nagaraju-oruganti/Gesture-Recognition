import os
import numpy as np
import pandas as pd
import random

# Image read and resize
from PIL import Image, ImageFilter, ImageEnhance
import cv2

class Generator:
    def __init__(self, config, data, **kwargs):
        self.config = config
        self.seed = config.seed
        self.augs = config.list_of_augmentations
        self.df = data.copy()
        self.batch_size = kwargs.get('batch_size', 32)
        self.shuffle = kwargs.get('shuffle', False)
        self.clips_dir = os.path.join(config.data_dir, 'raw')
        if config.colab_clips_dir is not None:
            self.clips_dir = config.colab_clips_dir
        self.kind = 'train' if self.shuffle else 'val'
    
    #-----------------------------------------------------------
    # HELPER METHODS
    #-----------------------------------------------------------
    @staticmethod
    def augment(image, augs, n = 1):
        '''Apply image filter
        '''
        n = min(len(augs) - 1, n)
        track = []
        for _ in range(n): 
            sample_aug = random.choice(augs if len(track) == 0 else [a for a in augs if a not in track])
            if sample_aug == 'EDGE_ENHANCEMENT':
                image = np.array(Image.fromarray(image, 'RGB').filter(ImageFilter.EDGE_ENHANCE))
            elif sample_aug == 'BLUR':
                image =  np.array(Image.fromarray(image, 'RGB').filter((ImageFilter.GaussianBlur(radius = 1))))
            elif sample_aug == 'DETAILING':
                image =  np.array(Image.fromarray(image, 'RGB').filter(ImageFilter.DETAIL))
            elif sample_aug == 'SHARPEN':
                image =  np.array(Image.fromarray(image, 'RGB').filter(ImageFilter.SHARPEN))
            elif sample_aug == 'BRIGHTEN':
                image =  np.array(ImageEnhance.Brightness(Image.fromarray(image, 'RGB')).enhance(1.5))
            else:
                image =  np.array(Image.fromarray(image, 'RGB'))
            track.append(sample_aug)
        return image
    
    @staticmethod
    def crop(image):
        '''Make rectangle image to square by cropping in the center'''
        w, h, c = image.shape       # height, width, channels
        if w > h:
            d = w - h
            image = image[:, d//2 : d//2 + h]
        elif w < h:
            d = h - w
            image = image[:, d//2 : d//2 + w]
        return image
    
    @staticmethod
    def normalize(image, method = 'normalize'):
        if method == 'normalize':
            # normalize by max pixel value
            image = image / 255.
        elif method == 'standardize':
            for i in [0, 1, 2]: 
                image[:, :, i] = (image[:, :, i] - np.mean(image[:, :, i])) / np.std(image[:, :, i])
        else:
            raise ValueError(f'invalid normalization `method` - {method} as passed as argument')
        return image
    
    def preprocess(self, folder_name, shall_augment):
        folder_path = os.path.join(self.clips_dir, self.kind, folder_name)
        clip = np.zeros((self.config.frames, self.config.height, self.config.width, self.config.channels), dtype = np.float32)
        frames = [file for i, file in enumerate(sorted(os.listdir(folder_path))) if i in self.config.frame_idx]
        for idx, img_name in enumerate(frames, start = 0):
            image = Image.open(f'{folder_path}/{img_name}')         # load image (image name includes extension)     
            image = image.resize((self.config.width, self.config.height)) # resize image to configured hxw
            image = np.array(image)

            if shall_augment:
                image = self.augment(image, augs = self.augs)       # apply random augmentation from selected typess
            image = self.normalize(image, method = self.config.normalization_method)
            
            clip[idx, :, :, :] = image                              # save in the clip array

        return clip
    
    #-----------------------------------------------------------
    # DATA LOADER
    #-----------------------------------------------------------
    def load(self):
        '''
            Dataframe contains
            - folder name
            - gesture description
            - corresponding label
            - shall augment (all False for validation)
        '''
        while True:
          ## Shuffle clips
          if self.shuffle:
              self.df = self.df.sample(frac = 1, random_state = self.seed)
          
          ## get clip dimensions
          n_frames, channels, height, width = self.config.clip_dim
          n_labels = self.config.n_labels
          num_batches = int(len(self.df) / self.batch_size) if len(self.df) % self.batch_size == 0 \
              else (len(self.df) // self.batch_size) + 1
          for batch_idx in range(0, num_batches):
              # placeholder to save clips in batches
              data = np.zeros((self.batch_size, n_frames, height, width, channels), dtype = np.float32)
              labels = np.zeros((self.batch_size, n_labels))
              n = 0
              for idx in range(0, self.batch_size):
                  clip_idx = idx + batch_idx * self.batch_size
                  if clip_idx >= len(self.df):
                      break   # data exhausted and exit while loop
                  
                  # make clip and corresponding clip for the batch)
                  [folder_name, _, label, shall_augment] = self.df.iloc[clip_idx].values
                  
                  # process clip
                  clip = self.preprocess(folder_name = folder_name, shall_augment = shall_augment)
                  
                  # save clip
                  data[idx, :, :, :, :] = clip
                  
                  labels[idx, label] = 1
                  n += 1

              # yield batch data
              yield data[:n, :, :, :, :], labels[:n, :] 