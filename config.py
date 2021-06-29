'''
Module :  config
Author:  Nasibullah (nasibullah104@gmail.com)
Details : Ths module consists of all hyperparameters and path details corresponding to Different models and datasets.
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
        self.dataset = 'msvd'; assert self.dataset in ['msvd','msrvtt']
        
        self.cuda_device_id = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')
        
        
        #data configuration
        self.batch_size = 200 #suitable 
        self.val_batch_size = 10
        self.max_caption_length = 30
        self.opt_truncate_caption = False
        self.vocabulary_min_count = 3
        
        #encoder configuration
        self.appearance_feature_extractor = 'inceptionv4'  
        self.appearance_input_size = 1536
        self.appearance_projected_size = 256
        self.opt_encoder = True
        
        #Decoder configuration
        self.decoder_type = 'lstm'
        self.embedding_size = 256
        self.decoder_input_size = self.embedding_size
        if self.opt_encoder: 
            self.decoder_hidden_size = self.appearance_projected_size
        else:
            self.decoder_hidden_size = self.appearance_input_size
        self.n_layers = 1
        self.dropout = 0.1
        self.opt_param_init = True   # manually sets parameter initialisation strategy
        
        #Training configuration
        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 0.7 #
        self.clip = 5 #clip the gradient to counter exploding gradient problem
        self.print_every = 400

        #vocabulary configuration
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
            self.feature_path = os.path.join(self.local_path,'features')
        
            self.name_mapping_file = os.path.join(self.caption_path,'youtube_mapping.txt')
            self.train_annotation_file = os.path.join(self.caption_path,'sents_train_lc_nopunc.txt')
            self.val_annotation_file = os.path.join(self.caption_path,'sents_val_lc_nopunc.txt')
            self.test_annotation_file = os.path.join(self.caption_path,'sents_test_lc_nopunc.txt')
            
          
            if cfg.appearance_feature_extractor == 'inceptionv4':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_INCEPTIONV4_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'inceptionresnetv2':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_INCEPTIONRESNETV2_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'resnet101':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_RESNET101_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'resnet101hc':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_RESNET101_HC.hdf5')
            
            self.motion_feature_file = os.path.join(self.feature_path,'MSVD_MOTION_RESNEXT101.hdf5')
            #self.object_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_INCEPTIONV4.hdf5')
                

        if cfg.dataset == 'msrvtt':
            self.local_path = os.path.join(working_path,'MSRVTT')
            self.video_path = '/media/nasibullah/Ubuntu/DataSets/MSRVTT/'
            self.caption_path = os.path.join(self.local_path,'captions')
            self.feature_path = os.path.join(self.local_path,'features')
            
            self.category_file_path = os.path.join(self.caption_path,'category.txt')
            self.train_val_annotation_file = os.path.join(self.caption_path,'train_val_videodatainfo.json')
            self.test_annotation_file = os.path.join(self.caption_path,'test_videodatainfo.json')
            
            if cfg.appearance_feature_extractor == 'inceptionv4':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSRVTT_APPEARANCE_INCEPTIONV4_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'inceptionresnetv2':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSRVTT_APPEARANCE_INCEPTIONRESNETV2_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'resnet101':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSRVTT_APPEARANCE_RESNET101_28.hdf5')
                
                
                

            self.val_id_list = list(range(6513,7010))
            self.train_id_list = list(range(0,6513))
            self.test_id_list = list(range(7010,10000))

        self.prediction_path = 'results'
        self.saved_models_path = 'Saved'
        
        
        
class ConfigS2VT:
    '''
    Hyperparameter settings for Sequence to Sequence, Video to text (S2VT) model.
    '''
    def __init__(self,model_name='s2vt'):
        self.model_name = model_name
        
        #Device configuration
        self.cuda_device_id = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')
        
        
        #Dataloader configuration
        self.dataset = 'msvd' # 'msvd' and 'msrvtt'
        self.batch_size = 100 
        self.val_batch_size = 10
        self.opt_truncate_caption = True
        self.max_caption_length = 30
        self.frame_len = 28
        
        # Encoder-decoder related configuration
        self.appearance_feature_extractor = 'inceptionv4'
        self.decoder_type = 'lstm'
        self.appearance_input_size = 1536
        self.appearance_projected_size = 500
        self.embedding_size = 500 # word embedding size
        self.max_length = 15
        
        self.decoder_l1_input_size = self.appearance_projected_size 
        self.decoder_l1_hidden_size = 1000
        self.decoder_l2_input_size = self.decoder_l1_hidden_size + self.embedding_size
        self.decoder_l2_hidden_size = 1000
        
        self.embed_dropout = 0.5
        self.rnn_l1_dropout = 0.4
        self.rnn_l2_dropout = 0.4
        
        self.opt_param_init = False   # manually sets parameter initialisation strategy
        self.beam_length = 5
        
        
        #Training configuration
        
        self.lr = 1e-4
        self.teacher_forcing_ratio = 1.0 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400
        
        
        #Vocabulary configuration
        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3
        self.vocabulary_min_count = 5
        
class ConfigSALSTM:
    '''
    Hyperparameter settings for Soft Attention based LSTM (SA-LSTM) model.
    '''
    def __init__(self,model_name='sa-lstm',opt_encoder=False):
        self.model_name = model_name
        
        #Device configuration
        self.cuda_device_id = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')
        
        #Dataloader configuration
        self.dataset = 'msvd' 
        self.batch_size = 100 #suitable 
        self.val_batch_size = 10
        self.opt_truncate_caption = True
        self.max_caption_length = 30
        
        
        #Encoder configuration
        self.appearance_feature_extractor = 'resnet101'  
        self.appearance_input_size = 2048
        self.appearance_projected_size = 512
        self.frame_len = 28
        self.opt_encoder = opt_encoder
        
        
        #Decoder configuration
        self.decoder_type = 'lstm'
        self.embedding_size = 468 # word embedding size
        if self.opt_encoder:
            self.feat_size =  self.appearance_projected_size
        else:
            self.feat_size = self.appearance_input_size 
         
        self.decoder_input_size = self.feat_size + self.embedding_size
        self.decoder_hidden_size = 512  #Hidden size of decoder LSTM
        self.attn_size = 128  # attention bottleneck
        self.n_layers = 1
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.4
        self.opt_param_init = False
        self.beam_length = 5
       
        
        #Training configuration
        
        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.teacher_forcing_ratio = 1.0 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400
        
        self.lr_decay_start_from = 20
        self.lr_decay_gamma = 0.5
        self.lr_decay_patience = 5
        self.weight_decay = 1e-5
        self.reg_lambda = 0.
        

        #Vocabulary configuration
        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3
        self.vocabulary_min_count = 5
        
        


class ConfigRecNet:
    '''
    Hyperparameter settings for Reconstruction (RecNet) model
    Beam search with beam length 5
    '''
    def __init__(self,model_name='recnet',opt_encoder=True):
        self.model_name = model_name
        
        #Device configuration
        self.cuda_device_id = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')
        
        #Dataloader configuration
        self.dataset = 'msvd' 
        self.batch_size = 100 #suitable 
        self.val_batch_size = 10
        self.opt_truncate_caption = True
        self.max_caption_length = 30
        
        
        #Encoder configuration
        self.appearance_feature_extractor = 'inceptionresnetv2'  
        self.appearance_input_size = 1536
        self.appearance_projected_size = 512
        self.frame_len = 28
        self.opt_encoder = opt_encoder
        
        
        #Decoder configuration
        self.decoder_type = 'lstm'
        if self.opt_encoder:
            self.feat_size =  self.appearance_projected_size
        else:
            self.feat_size = self.appearance_input_size
        
        self.embedding_size = 468 # word embedding size 
        self.decoder_input_size = self.feat_size + self.embedding_size
        self.decoder_hidden_size = 512  #Hidden size of decoder LSTM
        self.attn_size = 128  # attention bottleneck
        self.n_layers = 1
        self.dropout = 0.5
        self.rnn_dropout = 0.4
        self.opt_param_init = False
        self.beam_length = 5
       
        
        #Training configuration
        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.global_lr = 1e-4
        self.local_lr = 1e-4
        self.teacher_forcing_ratio = 1.0 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400
        self.lmda = 0.2
        self.training_stage = 1
        
        self.lr_decay_start_from = 20
        self.lr_decay_gamma = 0.5
        self.lr_decay_patience = 5
        self.weight_decay = 1e-5
        

        #Vocabulary related configuration
        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3
        self.vocabulary_min_count = 5
        
        
        #Global reconstructor configuration
        self.global_reconstructor_type = 'lstm'
        self.global_reconstructor_n_layers = 1
        self.global_reconstructor_hidden_size = self.feat_size    
        self.global_reconstructor_rnn_dropout = 0.4
        
        #Local reconstructor configuration
        self.local_reconstructor_type = 'lstm'
        self.local_reconstructor_n_layers = 1
        self.local_reconstructor_hidden_size = self.feat_size
        self.local_reconstructor_rnn_dropout = 0.4
        self.local_reconstructor_attn_size = 64
        
        self.reconstructor_type = 'local'
        
        
class ConfigMARN:
    '''
    Hyperparameter settings for MARN model.
    '''
    def __init__(self,model_name='marn'):
        
        self.model_name = model_name
        self.cuda_device_id = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')

        
        #Data related Configuration
        self.dataset = 'msvd' # from set {'msvd','msrvtt'}
        self.batch_size = 48 #suitable 
        self.val_batch_size = 10
        self.opt_truncate_caption = True
        self.max_caption_length = 30
        
        
        # Encoder related configuration
        self.appearance_feature_extractor = 'inceptionresnetv2'
        self.motion_feature_extractor = 'resnext101'
        self.frame_len = 28
        self.motion_depth = 16
        self.appearance_input_size = 1536
        self.appearance_projected_size = 512
        self.motion_input_size = 2048
        self.motion_projected_size = 512
        
        # Decoder related configuration
        self.feat_size = self.appearance_projected_size
        self.embedding_size = 512 # word embedding size
        self.decoder_input_size = self.appearance_projected_size + self.motion_projected_size + self.embedding_size
        self.decoder_type = 'lstm' # from set {lstm,gru}
        self.decoder_hidden_size = 512
        self.attn_size = 128
        self.n_layers = 1
        self.dropout = 0.5
        self.rnn_dropout = 0.4
        self.opt_param_init = False   # manually sets parameter initialisation strategy
        self.beam_length = 5
        
        # Training related configuration
        self.encoder_lr = 1e-4
        self.decoder_lr = 1e-4
        self.memory_decoder_lr = 1e-4
        self.acl_weight = 0.1
        self.teacher_forcing_ratio = 1.0
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400
        self.total_epochs = 1000
        self.lr_reduction = 0.5
        self.lr_reduction_step = 50
        

        #Vocabulary related configuration
        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3
        self.vocabulary_min_count = 5
        
        #Attend memory decoder configuration
        self.topk = 128
        self.amd_bottleneck_size = 64
        self.opt_memory_decoder = False
        self.lamb = 0.4
        
    def update(self):
        self.decoder_input_size = self.appearance_projected_size + self.motion_projected_size + self.embedding_size
            
  
