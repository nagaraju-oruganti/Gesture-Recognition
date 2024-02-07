### Configuration
class Config:
    
    # Seed for random number generation
    seed = 42
    
    # Repositories
    data_dir        = ''
    models_dir      = ''
    colab_clips_dir = None
    model_name      = ''
    
    # inputs and outputs
    target = 'label'
    class_weights = []
    n_labels = 5
    
    # Clip / image params
    clip_dim = (30, 3, 120, 120) #frames, channels, width, height
    normalization_method = 'normalize'
    
    # Augmentation parameters
    aug_size = 0
    list_of_augmentations = [
        'EDGE_ENHANCEMENT',
        'BLUR',
        'DETAILING',
        'SHARPEN',
        'BRIGHTEN'
    ]
    
    # Ablation study
    ablation_size = (None, None)    # train and validation size per label
                                    # None means all samples are considered for training
                                    
    # Train parameters
    num_epochs = 100
    train_batch_size = 4
    valid_batch_size = 8
    
    # Earlystopping parameters
    earlystopping_params = dict(
        min_delta = 0,
        patience = 10
    )
    
    # Scheduler parameters
    scheduler_params = dict(
        factor = 0.1,
        patience = 4,
        min_lr = 5e-5
    )
    
    # Misc parameters
    save_checkpoint = True
    