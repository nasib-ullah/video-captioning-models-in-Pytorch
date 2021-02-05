'''
Module :  config
Author:  Nasibullah (nasibullah104@gmail.com)
Details : Ths module consists of all hyperparameters and path details corresping to Different models and datasets.
          Only changing this module is enough to play with different model configurations. 
Use : Each model has its own configuration class which contains all hyperparameter and other settings. once Configuration object is created, use it to create Path object.
          
'''


import torch
import os

class ConfigMP:
    '''
    Hyperparameter settings for Mean Pooling model.
    '''
    def __init__(self,model_name='mean_pooling'):
        self.model_name = model_name
        self.dataset = 'msvd' # 'msvd' and 'msrvtt'
        self.encoder_arch = 'inceptionv4'  #vgg- densenet-1920 resnet inceptionv4 - 1536
        self.decoder_type = 'lstm'
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
        else:
            self.device = torch.device('cpu')

        self.batch_size = 64 #suitable 
        self.val_batch_size = 10
        if self.encoder_arch == 'inceptionv4':
            self.input_size = 1536
        if self.encoder_arch == 'densenet':
            self.input_size = 1920
        self.videofeat_size = 256
        self.embedding_size = 256 # word embedding size
        self.hidden_size = 256
        self.n_layers = 1
        self.dropout = 0.4

        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400

        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3

class Path:
    '''
    Currently supports MSVD and MSRVTT
    VATEX will be added in future
    '''
    def __init__(self,cfg,working_path):

        if cfg.dataset == 'msvd':   
            self.local_path = os.path.join(working_path,'MSVD')     
            self.video_path = 'path_to_raw_video_data' # For future use
            self.caption_path = os.path.join(self.local_path,'captions')
        
            self.name_mapping_file = os.path.join(self.caption_path,'youtube_mapping.txt')
            self.train_annotation_file = os.path.join(self.caption_path,'sents_train_lc_nopunc.txt')
            self.val_annotation_file = os.path.join(self.caption_path,'sents_val_lc_nopunc.txt')
            self.test_annotation_file = os.path.join(self.caption_path,'sents_test_lc_nopunc.txt')
            
            if cfg.encoder_arch == 'densenet':
            	self.feature_path = os.path.join(os.path.join(self.local_path,'features'),'Densenet')
            elif cfg.encoder_arch == 'vgg':
                self.feature_path = os.path.join(os.path.join(self.local_path,'features'),'VGG')
            else:
                self.feature_path = os.path.join(os.path.join(self.local_path,'features'),'Inceptionv4')
                self.feature_file = os.path.join(self.feature_path,'MSVD_InceptionV4.hdf5')
                

        if cfg.dataset == 'msrvtt':
            self.local_path = os.path.join(working_path,'MSRVTT')
            self.video_path = '/media/nasibullah/Ubuntu/DataSets/MSRVTT/'
            self.caption_path = os.path.join(self.local_path,'captions')
            self.feature_path = os.path.join(os.path.join(self.local_path,'features'),'Inceptionv4')

            self.category_file_path = os.path.join(self.caption_path,'category.txt')
            self.train_val_annotation_file = os.path.join(self.caption_path,'train_val_videodatainfo.json')
            self.test_annotation_file = os.path.join(self.caption_path,'test_videodatainfo.json')
            self.feature_file = os.path.join(self.feature_path,'MSR-VTT_InceptionV4.hdf5')

            self.val_id_list = list(range(6513,7010))
            self.train_id_list = list(range(0,6513))
            self.test_id_list = list(range(7010,10000))

        self.prediction_path = 'results'
        self.saved_models_path = 'Saved'


class ConfigSALSTM:
    '''
    Hyperparameter settings for Soft Attention based LSTM (SA-LSTM) model.
    '''
    def __init__(self,model_name='sa-lstm'):
        self.model_name = model_name
        self.dataset = 'msvd' # 'msvd' and 'msrvtt'
        self.encoder_arch = 'inceptionv4'  #vgg- densenet-1920 resnet inceptionv4 - 1536
        self.decoder_type = 'lstm'
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
        else:
            self.device = torch.device('cpu')

        self.batch_size = 100 #suitable 
        self.val_batch_size = 10
        if self.encoder_arch == 'inceptionv4':
            self.feat_size = 1536
        if self.encoder_arch == 'densenet':
            self.feat_size = 1920
        self.feat_len = 40
        self.embedding_size = 468 # word embedding size
        self.hidden_size = 512
        self.attn_size = 256
        self.n_layers = 1
        self.dropout = 0.4
        self.rnn_dropout = 0.1

        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400

        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3


class ConfigS2VT:
    '''
    Hyperparameter settings for Sequence to Sequence, Video to text (S2VT) model.
    '''
    def __init__(self,model_name='mean_pooling'):
        self.model_name = model_name
        self.dataset = 'msvd' # 'msvd' and 'msrvtt'
        self.encoder_arch = 'inceptionv4'  #vgg- densenet-1920 resnet inceptionv4 - 1536
        self.decoder_type = 'lstm'
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
        else:
            self.device = torch.device('cpu')

        self.batch_size = 64 #suitable 
        self.val_batch_size = 10
        if self.encoder_arch == 'inceptionv4':
            self.input_size = 1536
        if self.encoder_arch == 'densenet':
            self.input_size = 1920
        self.videofeat_size = 256
        self.embedding_size = 256 # word embedding size
        self.hidden_size = 256
        self.n_layers = 1
        self.dropout = 0.4

        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400

class ConfigRecNet:
    '''
    Hyperparameter settings for Reconstruction Network (RecNet) model.
    '''
    def __init__(self,model_name='mean_pooling'):
        self.model_name = model_name
        self.dataset = 'msvd' # 'msvd' and 'msrvtt'
        self.encoder_arch = 'inceptionv4'  #vgg- densenet-1920 resnet inceptionv4 - 1536
        self.decoder_type = 'lstm'
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
        else:
            self.device = torch.device('cpu')

        self.batch_size = 64 #suitable 
        self.val_batch_size = 10
        if self.encoder_arch == 'inceptionv4':
            self.input_size = 1536
        if self.encoder_arch == 'densenet':
            self.input_size = 1920
        self.videofeat_size = 256
        self.embedding_size = 256 # word embedding size
        self.hidden_size = 256
        self.n_layers = 1
        self.dropout = 0.4

        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400


class ConfigMARN:
    '''
    Hyperparameter settings for MARN model.
    '''
    def __init__(self,model_name='sa-lstm'):
        self.model_name = model_name
        self.dataset = 'msvd' # 'msvd' and 'msrvtt'
        self.encoder_arch = 'inceptionv4'  #vgg- densenet-1920 resnet inceptionv4 - 1536
        self.decoder_type = 'lstm'
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
        else:
            self.device = torch.device('cpu')

        self.batch_size = 100 #suitable 
        self.val_batch_size = 10
        if self.encoder_arch == 'inceptionv4':
            self.feat_size = 1536
        if self.encoder_arch == 'densenet':
            self.feat_size = 1920
        self.feat_len = 40
        self.embedding_size = 468 # word embedding size
        self.hidden_size = 512
        self.attn_size = 256
        self.n_layers = 1
        self.dropout = 0.4
        self.rnn_dropout = 0.1

        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400

        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3
