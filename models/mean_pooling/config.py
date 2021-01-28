import torch
import os

class Config:
    '''
    Currently Supported Models
    1. mean Pooling
    2. SA-LSTM
    3. S2VT
    4. RecNet
    '''
    def __init__(self,model_name='mean_pooling'):
        self.model_name = model_name
        self.encoder_arch = 'vgg'
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
        else:
            self.device = torch.device('cpu')

        self.batch_size = 64 #suitable 
        self.val_batch_size = 10
        self.embedding_size = 256 # word embedding size
        self.encoder_input_size = 1920
        self.hidden_size = 256
        self.decoder_n_layers = 2

        self.encoder_lr = 1e-5
        self.decoder_lr = 1e-3
        self.dropout = 0.4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400

class Path:
    '''
    Currently supports only MSVD
    MSRVTT will be added
    Also VATEX will be added in future
    '''
    def __init__(self,video_path='/media/nasibullah/Ubuntu/DataSets/MSVD/YouTube/'):
        
        self.video_path = video_path
        self.caption_path = os.path.join('MSVD','captions')
        self.feature_path = os.path.join(os.path.join('MSVD','features'),'Densenet')
        
        self.prediction_path = 'results'
        self.saved_models_path = 'Save'
        
        self.name_mapping_file = os.path.join(self.caption_path,'youtube_mapping.txt')
        self.train_annotation_file = os.path.join(self.caption_path,'sents_train_lc_nopunc.txt')
        self.val_annotation_file = os.path.join(self.caption_path,'sents_val_lc_nopunc.txt')
        self.test_annotation_file = os.path.join(self.caption_path,'sents_test_lc_nopunc.txt')
