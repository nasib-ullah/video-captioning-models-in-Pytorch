'''
Module :  RecNet model
Authors:  Nasibullah (nasibullah104@gmail.com)
'''


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as transforms

import random
import itertools
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import itertools

import numpy as np
import os
import copy
    

class Encoder(nn.Module):
    
    def __init__(self,cfg):
        super(Encoder,self).__init__()
        '''
        Encoder module. Project the video feature into a different space which will be 
        send to decoder.
        Argumets:
          input_size : CNN extracted feature size. For Densenet 1920, For inceptionv4 1536
          output_size : Dimention of projected space.
        '''
        
        self.appearance_projection_layer = nn.Linear(cfg.appearance_input_size,cfg.appearance_projected_size)
        
        
    def forward(self,x):
        appearance_out = self.appearance_projection_layer(x)
        
        return appearance_out
    

class TemporalAttention(nn.Module):
    def __init__(self,cfg):
        super(TemporalAttention,self).__init__()
        '''
        Spatial Attention module. It depends on previous hidden memory in the decoder(of shape hidden_size),
        feature at the source side ( of shape(196,feat_size) ).  
        at(s) = align(ht,hs)
              = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
        where
        score(ht,hs) = ht.t * hs                         (dot)
                     = ht.t * Wa * hs                  (general)
                     = va.t * tanh(Wa[ht;hs])           (concat)  
        Here we have used concat formulae.
        Argumets:
          hidden_size : hidden memory size of decoder. (batch,hidden_size)
          feat_size : feature size of each grid (annotation vector) at encoder side.
          bottleneck_size : intermediate size.
        '''
        
        
        self.hidden_size = cfg.decoder_hidden_size
        self.feat_size = cfg.feat_size
        self.bottleneck_size = cfg.attn_size
        
        self.decoder_projection = nn.Linear(self.hidden_size,self.bottleneck_size)
        self.encoder_projection = nn.Linear(self.feat_size, self.bottleneck_size)
        self.final_projection = nn.Linear(self.bottleneck_size,1)
     
    def forward(self,hidden,feats):
        '''
        shape of hidden (hidden_size) (batch,hidden_size) #(100,512)
        shape of feats (batch size,feat_len,feat_size)  #(100,40,1536)
        '''
        Wh = self.decoder_projection(hidden)  
        Uv = self.encoder_projection(feats)   
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.final_projection(torch.tanh(Wh+Uv))
        weights = F.softmax(energies, dim=1)
        weighted_feats = feats *weights.expand_as(feats)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats,weights


class DecoderRNN(nn.Module):
    
    def __init__(self,cfg,voc):
        super(DecoderRNN, self).__init__()
        '''
        Decoder, Basically a language model.
        Args:
        hidden_size : hidden memory size of LSTM/GRU
        output_size : output size. Its same as the vocabulary size.
        n_layers : 
        
        '''
        
        # Keep for reference
        
        
        # Keep for reference
        self.dropout = cfg.dropout
        self.feat_len = cfg.frame_len
        self.attn_size = cfg.attn_size
        self.output_size = voc.num_words
        self.rnn_dropout = cfg.rnn_dropout
        self.n_layers = cfg.n_layers
        self.decoder_type = cfg.decoder_type

        # Define layers
        self.embedding = nn.Embedding(voc.num_words, cfg.embedding_size)
        self.attention = TemporalAttention(cfg)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        if self.decoder_type == 'gru':
            self.rnn = nn.GRU(input_size=cfg.decoder_input_size, hidden_size=cfg.decoder_hidden_size,
                              num_layers=self.n_layers, dropout=self.rnn_dropout)
        else:
            self.rnn = nn.LSTM(input_size=cfg.decoder_input_size, hidden_size=cfg.decoder_hidden_size,
                           num_layers=self.n_layers, dropout=self.rnn_dropout)
        self.out = nn.Linear(cfg.decoder_hidden_size, self.output_size)

    
    def forward(self, inputs, hidden, feats):
        '''
        we run this one step (word) at a time
        
        inputs -  (1, batch)
        hidden - h_n/c_n :(num_layers * num_directions, batch, hidden_size)    # GRU:h_n   LSTM:(h_n,c_n)
        feats - (batch,attention_length,annotation_vector_size) 
        
        '''
        embedded = self.embedding(inputs) # [i/p:(1,batch)  o/p:(1,batch,embedding_size)]
        last_hidden = hidden[0] if self.decoder_type=='lstm' else hidden
        last_hidden = last_hidden.view(self.n_layers,last_hidden.size(1),last_hidden.size(2))
        last_hidden = last_hidden[-1]
        feats, attn_weights = self.attention(last_hidden,feats) #(100,1536) #(100,28,1)
        input_combined = torch.cat((embedded,feats.unsqueeze(0)),dim=2)
        output, hidden = self.rnn(input_combined, hidden) # (1,100,512)
        output = output.squeeze(0) # (100,512)
        output = self.out(output) # (100,num_words)
        output = F.softmax(output, dim = 1) #(100,num_words)
        return output, hidden, attn_weights 
    
    
class GlobalReconstructor(nn.Module):
    def __init__(self,cfg):
        super(GlobalReconstructor,self).__init__()
        
        self.num_layers = cfg.global_reconstructor_n_layers
        self.input_size = cfg.decoder_hidden_size *2
        self.hidden_size = cfg.global_reconstructor_hidden_size
        self.rnn_dropout = cfg.global_reconstructor_rnn_dropout
        self.decoder_type = cfg.global_reconstructor_type
        
        if cfg.global_reconstructor_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_size,hidden_size = self.hidden_size,
                              num_layers = self.num_layers,dropout=self.rnn_dropout)
        
        if cfg.global_reconstructor_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size,hidden_size = self.hidden_size,
                              num_layers = self.num_layers,dropout=self.rnn_dropout)
        
    def forward(self,decoder_hidden,decoder_hiddens_mean_pooled, hidden):
        '''
        Args:
            decoder_hidden : (1,batch_size,512)
            decoder_hiddens_mean_pooled : (1,batch_size,512)
            hidden : (1,batch_size,1536)
            
        Note : For LSTM input : (seq_len,batch,input_size) here (1,B,1024)
                        hidden : (n_layers*n_directions,batch_size,hidden_size)
                        output : (seq_len,batch,num_directions*hidden_size) here (1,B,1536)
        '''
        #print(decoder_hidden.size(),decoder_hiddens_mean_pooled.size())
        input_combined = torch.cat([decoder_hidden,decoder_hiddens_mean_pooled],dim=2)
        #input_combined = input_combined.unsqueeze(0)

        output,hidden = self.rnn(input_combined,hidden)

        return output,hidden
    
    
class ReconstructorAttention(nn.Module):
    def __init__(self,cfg):
        super(ReconstructorAttention,self).__init__()
        '''
        Temporal Attention module. It depends on previous hidden memory in the local reconstructor
        (of shape hidden_size),
        feature at the source side ( of shape(196,feat_size) ).  
        at(s) = align(ht,hs)
              = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
        where
        score(ht,hs) = ht.t * hs                         (dot)
                     = ht.t * Wa * hs                  (general)
                     = va.t * tanh(Wa[ht;hs])           (concat)  
        Here we have used concat formulae.
        Argumets:
          hidden_size : hidden memory size of decoder. (batch,hidden_size)
          feat_size : feature size of each grid (annotation vector) at encoder side.
          bottleneck_size : intermediate size.
        '''
        
        
        self.hidden_size = cfg.local_reconstructor_hidden_size
        self.feat_size = cfg.decoder_hidden_size
        self.bottleneck_size = cfg.local_reconstructor_attn_size
        
        self.decoder_projection = nn.Linear(self.hidden_size,self.bottleneck_size)
        self.encoder_projection = nn.Linear(self.feat_size, self.bottleneck_size)
        self.final_projection = nn.Linear(self.bottleneck_size,1)
     
    def forward(self,hidden,feats):
        '''
        shape of hidden (hidden_size) (batch,hidden_size) #(100,1536)
        shape of feats (batch size,feat_len,feat_size)  #(100,feat_len(decoder_time_step,varry),512) (T,100,512)
        '''
        feats = feats.permute(1,0,2)
        Wh = self.decoder_projection(hidden)  # (100,256)
        Uv = self.encoder_projection(feats)  # (100,15,256)
        #print('Wh :',Wh.size())
        #print('Uv :',Uv.size())
        
        Wh = Wh.unsqueeze(1).expand_as(Uv) # (100,15,256)
        energies = self.final_projection(torch.tanh(Wh+Uv)) # (100,15,1)
        #print('energies :',energies.size())
        weights = F.softmax(energies, dim=1) # (100,15,1)
        #print('weights :',weights.size()) 
        weighted_feats = feats *weights.expand_as(feats) #
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats,weights
    
    
class LocalReconstructor(nn.Module):
    def __init__(self,cfg):
        super(LocalReconstructor,self).__init__()
        
        self.num_layers = cfg.local_reconstructor_n_layers
        self.input_size = cfg.decoder_hidden_size
        self.hidden_size = cfg.local_reconstructor_hidden_size 
        self.rnn_dropout = cfg.local_reconstructor_rnn_dropout
        self.decoder_type = cfg.local_reconstructor_type 
        self.n_layers = 1
        
        self.attention = ReconstructorAttention(cfg)
        if self.decoder_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.rnn_dropout)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, dropout=self.rnn_dropout)
            
    
    def forward(self,decoder_hidden,hidden):
        '''
        Args:
            decoder_hidden : (batch,seq_length,decoder_hidden_size)
            
        Return:
            output : (batch,1536)
        
        
        '''
        
        last_hidden = hidden[0] if self.decoder_type=='lstm' else hidden
        last_hidden = last_hidden.view(self.n_layers,last_hidden.size(1),last_hidden.size(2))
        last_hidden = last_hidden[-1]
        feats, attn_weights = self.attention(last_hidden,decoder_hidden) #(100,512) #(100,28,1)
        input_combined = feats.unsqueeze(0)
        output, hidden = self.rnn(input_combined, hidden) # (1,100,512)
        output = output.squeeze(0) # (100,1536)
        
        return output, hidden, attn_weights 
    
        
        
class RecNet(nn.Module):
    def __init__(self,voc,cfg,path):
        super(RecNet,self).__init__()
        
        self.voc = voc
        self.cfg = cfg
        self.path = path
        
        if cfg.opt_encoder:
            self.encoder = Encoder(cfg).to(cfg.device)
            self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr) #change to AdaDelta
                                            
        self.decoder = DecoderRNN(cfg,voc).to(cfg.device)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr, amsgrad=True) # Adadelta
                                            
        
        self.reconstructor_type = cfg.reconstructor_type
        
        if self.reconstructor_type == 'global':
            self.global_reconstructor = GlobalReconstructor(cfg).to(cfg.device)
            self.global_optimizer = optim.Adam(self.global_reconstructor.parameters(),lr=cfg.global_lr, amsgrad=True) #Adadelta
            
                                            
        if self.reconstructor_type == 'local':
            self.local_reconstructor = LocalReconstructor(cfg).to(cfg.device)
            self.local_optimizer = optim.Adam(self.local_reconstructor.parameters(),lr=cfg.local_lr, amsgrad=True) #Adadelta
            
            
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.lmda = cfg.lmda
        self.print_every = cfg.print_every
        self.clip = cfg.clip
        self.device = cfg.device
        if cfg.opt_param_init:
            self.init_params()
            
        self.epoch = 0
            
    def init_params(self):
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
                
    def update_hyperparameters(self,cfg):
        
        if self.cfg.opt_encoder:
            self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)
        
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr,amsgrad=True)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.cfg.training_stage = cfg.training_stage
        
        if self.cfg.training_stage == 2:
            self.cfg.lmda = cfg.lmda
            if self.reconstructor_type == 'global':
                self.global_optimizer = optim.Adam(self.global_reconstructor.parameters(),lr=cfg.global_lr, amsgrad=True)

            if self.reconstructor_type == 'local':
                self.local_optimizer = optim.Adam(self.local_reconstructor.parameters(),lr=cfg.local_lr, amsgrad=True)

    
    def load(self,epoch):
        encoder_path = os.path.join(self.path.saved_models_path,'encoder_'+str(epoch)+'.pt')
        decoder_path = os.path.join(self.path.saved_models_path,'decoder_'+str(epoch)+'.pt')
        global_reconstructor_path = os.path.join(self.path.saved_models_path,'Global_'+str(epoch)+'.pt')
        local_reconstructor_path = os.path.join(self.path.saved_models_path,'local_'+str(epoch)+'.pt')
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')
            
        if self.reconstructor_type == 'global':
            try:
                self.global_reconstructor.load_state_dict(torch.load(global_reconstructor_path))
            except:
                print('File not found Error..')
                
        if self.reconstructor_type == 'local':
            try:
                self.local_reconstructor.load_state_dict(torch.load(local_reconstructor_path))
            except:
                print('File not found Error..')
        
        
    def save(self,encoder_path,decoder_path):
        
        encoder_path = os.path.join(self.path.saved_models_path,'encoder_'+str(self.epoch)+'.pt')
        decoder_path = os.path.join(self.path.saved_models_path,'decoder_'+str(self.epoch)+'.pt')
        global_reconstructor_path = os.path.join(self.path.saved_models_path,'Global_'+str(self.epoch)+'.pt')
        local_reconstructor_path = os.path.join(self.path.saved_models_path,'local_'+str(self.epoch)+'.pt')
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            
            torch.save(self.encoder.state_dict(),encoder_path)
            torch.save(self.decoder.state_dict(),decoder_path)
        else:
            print('Invalid path address given.')
            
        
        if self.reconstructor_type == 'global':
            try:
                torch.save(self.global_reconstructor.state_dict(),global_reconstructor_path)
            except:
                print('Invalid path address given.')
                
        if self.reconstructor_type == 'local':
            try:
                torch.save(self.local_reconstructor.state_dict(),local_reconstructor_path)
            except:
                print('Invalid path address given.')
                
                
    def forward_global_reconstructor(self,video_features,decoder_hidden):
        '''
        Args:
            video_features : (B,T,F)
            decoder_hidden : (T,B,h)
        
        Return:
            mean_global_output : (B,F) (100,1536)
            mean_video_feature : (B,F) (100,1536)
        
        '''
        global_output = []
        decoder_time_step = len(decoder_hidden)
        global_hidden = torch.zeros(1,self.cfg.batch_size,
                                    self.cfg.global_reconstructor_hidden_size).to(self.device)
        if self.cfg.global_reconstructor_type == 'lstm':
            global_hidden = (global_hidden,global_hidden)
        mean_video_feature = torch.mean(video_features,dim=1) #(B,F)
        mean_decoder_hidden = torch.mean(decoder_hidden,dim=0).unsqueeze(0) #(B,h) should be (1,B,512)
        for l in range(decoder_time_step):
            #print('forward global : ',decoder_hidden[l].size()) # should be (1,B,512)
            output,global_hidden = self.global_reconstructor(decoder_hidden[l].unsqueeze(0),mean_decoder_hidden,global_hidden)
            
            if self.cfg.global_reconstructor_type == 'gru':
                global_output.append(output.squeeze(0))
            if self.cfg.global_reconstructor_type == 'lstm':
                global_output.append(output[0].squeeze(0))
            
            
        global_output = torch.stack(global_output,0) # (T,B,1536) (T,B,F)
        mean_global_output = global_output.mean(dim=0) # (B,F)
        
        return mean_global_output, mean_video_feature
    
    def forward_local_reconstructor(self,video_features,decoder_hidden):
        '''
        Args:
            video_features : (B,T,F)
            decoder_hidden : (T,B,h)
            
        Return:
            video_features : (B,T,F)
            reconstruction_features : (B,T,F)
        
        '''
        
        local_output = []
        decoder_time_step = len(decoder_hidden)
        local_hidden = torch.zeros(1,self.cfg.batch_size,
                                   self.cfg.local_reconstructor_hidden_size).to(self.device)
        if self.cfg.global_reconstructor_type == 'lstm':
            local_hidden = (local_hidden,local_hidden)
        for l in range(self.cfg.frame_len):
            #print(decoder_hidden[l].size()) # should be (1,B,512)
            local_out,local_hidden,_ = self.local_reconstructor(decoder_hidden,local_hidden)
            #print('local_output :',local_out.size())
            local_output.append(local_out)
            
        local_output = torch.stack(local_output,0)
        
        return local_output,video_features
    
    def reconstruction_loss_calculate(self,predicted_features,gt_features):
        '''
        Generic for both local and global reconstructor
        Args :
            predicted_features : for global -> (B,F), for local (B,T,F)
            gt_features : for global -> (B,F), for local (B,T,F)
        
        '''
        rec_loss = 0
        if self.reconstructor_type == 'global':
            rec_loss = F.mse_loss(predicted_features,gt_features).mean()
            
        if self.reconstructor_type == 'local':
            #print('predicted :',predicted_features.size())
            #print('Gt :',gt_features.size())
            rec_loss = F.mse_loss(predicted_features.permute(1,0,2),gt_features).mean()
            
        #print('rec loss in loss calculate :',rec_loss)
            
        return rec_loss
    

    def train_epoch(self,dataloader,utils):
        '''
        Function to train the model for a single epoch.
        Args:
         Input:
            dataloader : the dataloader object.basically train dataloader object.
         Return:
             epoch_loss : Average single time step loss for an epoch
        '''
        total_lloss = 0
        total_recloss = 0
        start_iteration = 1
        print_lloss = 0
        print_recloss = 0
        iteration = 1
        if self.cfg.opt_encoder:
            self.encoder.train()
        self.decoder.train()
        if self.cfg.training_stage == 2:
            if self.reconstructor_type == 'global':
                self.global_reconstructor.train()
            else:
                self.local_reconstructor.train()
            
        for data in dataloader:
            features, targets, mask, max_length, _,_,_ = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            loss,rloss = self.train_iter(utils,features,targets,mask,max_length,use_teacher_forcing)
            print_lloss += loss
            print_recloss += rloss
            total_lloss += loss
            total_recloss += rloss
        # Print progress
            if iteration % self.print_every == 0:
                print_lloss_avg = print_lloss / self.print_every
                print_recloss_avg = print_recloss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average Likelihood loss: {:.4f}; Average Reconstruction loss: {:.4f}".format(iteration, iteration / len(dataloader) * 100, print_loss_avg,print_recloss_avg))
                print_loss = 0
             
            iteration += 1 
        return total_lloss/len(dataloader),total_recloss/len(dataloader)
    
    
    def train_iter(self,utils,input_variable, target_variable, mask,max_target_len,use_teacher_forcing):
        '''
        Forward propagate input signal and update model for a single iteration. 
        
        Args:
        Inputs:
            input_variable : video mini-batch tensor; size = (B,T,F)
            target_variable : Ground Truth Captions;  size = (T,B); T will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T,C)
            max_target_len : maximum lengh of the mini-batch; size = T
            use_teacher_forcing : binary variable. If True training uses teacher forcing else sampling.
            clip : clip the gradients to counter exploding gradient problem.
        Returns:
            iteration_loss : average loss per time step.
        '''
        if self.cfg.opt_encoder:
            self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        if self.reconstructor_type == 'global':
            self.global_optimizer.zero_grad() 
        if self.reconstructor_type == 'local':
            self.local_optimizer.zero_grad()
        
        
        likelihood_loss = 0
        total_loss = 0
        rec_loss = 0
        print_losses = []
        decoder_hidden_list = []
        n_totals = 0
        
        input_variable = input_variable.to(self.device)
        
        if self.cfg.opt_encoder:
            input_variable = self.encoder(input_variable)  
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        # Forward pass through encoder
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(self.cfg.batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, self.cfg.batch_size,
                                      self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
            
        
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,_ = self.decoder(decoder_input, decoder_hidden,input_variable.float())
                # Teacher forcing: next input comes from ground truth(data distribution)
                if self.cfg.decoder_type == 'gru':
                    decoder_hidden_list.append(decoder_hidden.squeeze(0))
                else:
                    decoder_hidden_list.append(decoder_hidden[0].squeeze(0))
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t],self.device)
                likelihood_loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,_ = self.decoder(decoder_input, decoder_hidden,input_variable.float())
                # No teacher forcing: next input is decoder's own current output(model distribution)
                _, topi = decoder_output.squeeze(0).topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.cfg.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                if self.cfg.decoder_type == 'gru':
                    decoder_hidden_list.append(decoder_hidden.squeeze(0))
                else:
                    decoder_hidden_list.append(decoder_hidden[0].squeeze(0))
                # Calculate and accumulate loss
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output, target_variable[t], mask[t],self.device)
                likelihood_loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
                
                

        # Perform backpropatation
        if self.cfg.training_stage == 1:
            likelihood_loss.backward()
        if self.cfg.training_stage == 2:
            decoder_hidden_list = torch.stack(decoder_hidden_list,0) #(T,B,h)
            #print('input variable :',input_variable.size())
            #print('decoder_hidden_list :',decoder_hidden_list.size())
            
            if self.cfg.reconstructor_type == 'global':
                pred_val,gt_value =  self.forward_global_reconstructor(input_variable,decoder_hidden_list)
                rec_loss = self.reconstruction_loss_calculate(pred_val,gt_value)

            if self.cfg.reconstructor_type == 'local':
                pred_val,gt_value = self.forward_local_reconstructor(input_variable,decoder_hidden_list)
                rec_loss = self.reconstruction_loss_calculate(pred_val,gt_value)
                
            total_loss = likelihood_loss + self.lmda*rec_loss
            total_loss.backward()
            rec_loss = rec_loss.item()

        if self.cfg.opt_encoder:
            _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            self.enc_optimizer.step()
        
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        self.dec_optimizer.step()
        
        if self.cfg.training_stage == 2:
            if self.cfg.reconstructor_type == 'global':
                _ = nn.utils.clip_grad_norm_(self.global_reconstructor.parameters(), self.clip)
                self.global_optimizer.step()
            if self.cfg.reconstructor_type == 'local':
                _ = nn.utils.clip_grad_norm_(self.local_reconstructor.parameters(), self.clip)
                self.local_optimizer.step()
        
        
        return sum(print_losses) / n_totals, rec_loss
    
    @torch.no_grad()
    def GreedyDecoding(self,features,max_length=15):
        batch_size = features.size()[0]
        features = features.to(self.device)
        
        if self.cfg.opt_encoder:
            features = self.encoder(features) #need to make optional
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(batch_size)]]).to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, batch_size,
                                      self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        caption = []
        attention_values = []
        for _ in range(max_length):
            decoder_output, decoder_hidden,attn_values = self.decoder(decoder_input, 
                                                            decoder_hidden,features.float())
            _, topi = decoder_output.squeeze(0).topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(self.device)
            caption.append(topi.squeeze(1).cpu())
            attention_values.append(attn_values.squeeze(2))
        caption = torch.stack(caption,0).permute(1,0)
        caps_text = []
        for dta in caption:
            tmp = []
            for token in dta:
                if token.item() not in self.voc.index2word.keys() or token.item()==2: # Remove EOS and bypass OOV
                    pass
                else:
                    tmp.append(self.voc.index2word[token.item()])
            tmp = ' '.join(x for x in tmp)
            caps_text.append(tmp)
        return caption,caps_text, torch.stack(attention_values,0).cpu().numpy()
    
    @torch.no_grad()
    def BeamDecoding(self,feats, width, alpha=0.,max_caption_len = 15):
        batch_size = feats.size(0)
        vocab_size = self.voc.num_words
        

        vfunc = np.vectorize(lambda t: self.voc.index2word[t]) # to transform tensors to words
        rfunc = np.vectorize(lambda t: '' if t == 'EOS' else t) # to transform EOS to null string
        lfunc = np.vectorize(lambda t: '' if t == 'SOS' else t) # to transform SOS to null string
        pfunc = np.vectorize(lambda t: '' if t == 'PAD' else t) # to transform PAD to null string
        
        if self.cfg.opt_encoder:
            feats = self.encoder(feats) 

        hidden = torch.zeros(self.cfg.n_layers, batch_size, self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            hidden = (hidden,hidden)
        
        input_list = [ torch.cuda.LongTensor(1, batch_size).fill_(self.cfg.SOS_token) ]
        hidden_list = [ hidden ]
        cum_prob_list = [ torch.ones(batch_size).cuda() ]
        cum_prob_list = [ torch.log(cum_prob) for cum_prob in cum_prob_list ]
        EOS_idx = self.cfg.EOS_token

        output_list = [ [[]] for _ in range(batch_size) ]
        for t in range(max_caption_len + 1):
            beam_output_list = [] # width x ( 1, 100 )
            normalized_beam_output_list = [] # width x ( 1, 100 )
            if self.cfg.decoder_type == "lstm":
                beam_hidden_list = ( [], [] ) # 2 * width x ( 1, 100, 512 )
            else:
                beam_hidden_list = [] # width x ( 1, 100, 512 )
            next_output_list = [ [] for _ in range(batch_size) ]
            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                output, next_hidden, _ = self.decoder(input, hidden, feats) # need to check

                caption_list = [ output_list[b][i] for b in range(batch_size)]
                EOS_mask = [ 0. if EOS_idx in [ idx.item() for idx in caption ] else 1. for caption in caption_list ]
                EOS_mask = torch.cuda.FloatTensor(EOS_mask)
                EOS_mask = EOS_mask.unsqueeze(1).expand_as(output)
                output = EOS_mask * output

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t + 1 for caption in caption_list ]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** alpha) / (6 ** alpha)
                normalizing_factor = normalizing_factor.unsqueeze(1).expand_as(output)
                normalized_output = output / normalizing_factor
                normalized_beam_output_list.append(normalized_output)
                if self.cfg.decoder_type == "lstm":
                    beam_hidden_list[0].append(next_hidden[0])
                    beam_hidden_list[1].append(next_hidden[1])
                else:
                    beam_hidden_list.append(next_hidden)
            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] # width * ( 100, )
            if self.cfg.decoder_type == "lstm":
                topk_hidden_list = (
                    [ [] for _ in range(width) ],
                    [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
            else:
                topk_hidden_list = [ [] for _ in range(width) ] # width * ( 1, 100, 512 )
            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    if self.cfg.decoder_type == "lstm":
                        topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                        topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    else:
                        topk_hidden_list[k].append(beam_hidden_list[bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            output_list = next_output_list

            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            if self.cfg.decoder_type == "lstm":
                hidden_list = (
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0] ],
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1] ]) # 2 * width * ( 1, 100, 512 )
                hidden_list = [ ( hidden, context ) for hidden, context in zip(*hidden_list) ]
            else:
                hidden_list = [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list ] # width * ( 1, 100, 512 )
            cum_prob_list = [ torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

        SOS_idx = self.cfg.SOS_token
        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        
        outputs = [[torch.tensor(y) for y in x] for x in outputs]
        outputs = [[y.item() for y in x] for x in outputs]
        
        captions = vfunc(outputs)
        captions = rfunc(captions)
        captions = lfunc(captions)
        captions = pfunc(captions)
        caps_text = []

        for eee in captions:
            caps_text.append(' '.join(x for x in eee).strip())
        
        
        return caps_text
    
