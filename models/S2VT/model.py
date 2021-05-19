'''
Module :  S2VT model
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
    
    

class EncoderDecoder(nn.Module):
    
    def __init__(self,cfg,voc):
        super(EncoderDecoder, self).__init__()
        '''
        Encoder decoder model.
        Args:
        hidden_size : hidden memory size of LSTM/GRU
        output_size : output size. Its same as the vocabulary size.
        n_layers : 
        
        '''
        
        # Keep for reference
        self.feat_len = cfg.frame_len
        self.caption_len = cfg.max_caption_length
        self.total_time_steps = self.feat_len + self.caption_len
        self.output_size = voc.num_words
        self.rnn_l1_dropout = cfg.rnn_l1_dropout
        self.rnn_l2_dropout = cfg.rnn_l2_dropout
        self.decoder_type = cfg.decoder_type
        self.l1_hidden = cfg.decoder_l1_hidden_size
        self.l2_hidden = cfg.decoder_l2_hidden_size
        self.device = cfg.device
        self.cfg = cfg

        # Define layers
        self.encoding = nn.Linear(cfg.appearance_input_size,cfg.appearance_projected_size)
        self.embedding = nn.Embedding(voc.num_words, cfg.embedding_size)
        self.embedding_dropout = nn.Dropout(cfg.embed_dropout)
        if self.decoder_type == 'gru':
            self.rnn1 = nn.GRU(input_size=cfg.decoder_l1_input_size, hidden_size=cfg.decoder_l1_hidden_size,
                              num_layers=1, dropout=self.rnn_l1_dropout)
            
            self.rnn2 = nn.GRU(input_size=cfg.decoder_l2_input_size, hidden_size=cfg.decoder_l2_hidden_size,
                              num_layers=1, dropout=self.rnn_l2_dropout)
        else:
            self.rnn1 = nn.LSTM(input_size=cfg.decoder_l1_input_size, hidden_size=cfg.decoder_l1_hidden_size,
                           num_layers=1, dropout=self.rnn_l1_dropout)
            
            self.rnn2 = nn.LSTM(input_size=cfg.decoder_l2_input_size, hidden_size=cfg.decoder_l2_hidden_size,
                           num_layers=1, dropout=self.rnn_l2_dropout)
            
        self.out = nn.Linear(cfg.decoder_l2_hidden_size, self.output_size)

    
    def forward(self,feats,target=None,use_teacher_forcing=False,training=True):
        '''
        we run this one step (word) at a time
        
        feats - (attention_length,batch,annotation_vector_size)
        target - (time_step,batch)
        use_teacher_forcing - boolean value. False during inference.
        
        we run the first lstm layer at once. 
        first half of 2nd layer we run at once.
        second half of second layer run through a loop to generate words.
        
        '''
        output = []
        feats = self.encoding(feats)
        if training:
            T = target.size()[0]
        else:
            T = self.cfg.max_length
        B, Ft = feats.size()[1], feats.size()[-1] 
        pad_tensor = torch.zeros(T,B,Ft).to(self.device)
        feats = torch.cat([feats,pad_tensor]).to(self.device)
        
        if self.decoder_type=='lstm':
            h1,c1 = torch.zeros(1,B,self.l1_hidden).to(self.device), torch.zeros(1,B,self.l1_hidden).to(self.device)
            h2,c2 = torch.zeros(1,B,self.l2_hidden).to(self.device), torch.zeros(1,B,self.l2_hidden).to(self.device)
            out1,(h1,c1) = self.rnn1(feats,(h1,c1))
        else:
            h1 = torch.zeros(1,B,self.l1_hidden).to(self.device)
            h2 = torch.zeros(1,B,self.l2_hidden).to(self.device)
            out1,h1 = self.rnn1(feats,h1) #out1 : (28+T,B,h)
        
        
        #print('out1 shape :',out1.size())
        sos_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(B)]]).to(self.device) #(1,B)
        
        if use_teacher_forcing:
            word_input = target[:-1] # (T-1,B)
            combined_input = torch.cat([sos_input,word_input]) 
            embedded = self.embedding(combined_input) # [i/p:(T,B)  o/p:(T,B,Emb)]
            dummy = torch.zeros(self.cfg.frame_len,B,self.cfg.embedding_size).to(self.device)
            embedded = torch.cat([dummy,embedded]) # (T+28,B,E)
            embedded = self.embedding_dropout(embedded)
            l1_output = torch.cat([out1,embedded],dim=2) # (28+T,B,h+E)
        else:
            #add only first 28 time steps.
            dummy = torch.zeros(self.cfg.frame_len,B,self.cfg.embedding_size).to(self.device) # (28,B,E)
            out_p1 = out1[:self.cfg.frame_len]
            l1_output = torch.cat([out_p1,dummy],dim=2) # (28,B,h+E) need to fix
            
        #pass 1st half one step. we don't really care about out2, only we care about is (h2,c2)/h2
        if self.decoder_type == 'lstm':
            out2,(h2,c2) = self.rnn2(l1_output[:self.cfg.frame_len],(h2,c2)) 
        else:
            out2,h2 = self.rnn2(l1_output[:self.cfg.frame_len],h2)
        
        #final decoding loop
        if use_teacher_forcing:
            for t in range(T):
                decoder_input = l1_output[self.cfg.frame_len+t].unsqueeze(0)
                if self.decoder_type == 'lstm':
                    decoder_output,(h2,c2) = self.rnn2(decoder_input,(h2,c2))
                else:
                    decoder_output,h2 = self.rnn2(decoder_input,h2)
                    
                decoder_output = self.out(decoder_output).squeeze(0)
                decoder_output = F.softmax(decoder_output, dim = 1) #(100,num_words)
                # Teacher forcing: next input comes from ground truth(data distribution)
                output.append(decoder_output.unsqueeze(0))
        else:
            word_emb = self.embedding(sos_input)
            for t in range(T):
                
                decoder_input = torch.cat([out1[self.cfg.frame_len+t].unsqueeze(0),word_emb],dim=2) #(1,B,E),(1,B,h) -> (1,B,E+h)
                
                if self.decoder_type == 'lstm':
                    decoder_output, (h2,c2) = self.rnn2(decoder_input,(h2,c2))
                else:
                    decoder_output, h2 = self.rnn2(decoder_input,h2)
                decoder_output = self.out(decoder_output)    
                decoder_output = F.softmax(decoder_output, dim = 1) #(100,num_words)
                
                # No teacher forcing: next input is decoder's own current output(model distribution)
                _, topi = decoder_output.squeeze(0).topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(B)]])
                decoder_input = decoder_input.to(self.device)
                word_emb = self.embedding(decoder_input)
                if training:
                    output.append(decoder_output)
                else:
                    output.append(topi.squeeze(1).cpu())
                
        output = torch.stack(output,0)
        
        return output  
    
    
class S2VT(nn.Module):
    
    def __init__(self,voc,cfg,path):
        super(S2VT,self).__init__()

        self.voc = voc
        self.path = path
        self.cfg = cfg
    
        self.encoderdecoder = EncoderDecoder(cfg,voc).to(cfg.device)
        self.encdec_optimizer = optim.Adam(self.encoderdecoder.parameters(),lr=cfg.lr)
    
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.print_every = cfg.print_every
        self.clip = cfg.clip
        self.device = cfg.device
        if cfg.opt_param_init:
            self.init_params()
        
        
    def init_params(self):
        for name, param in self.encoderdecoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

        
    def update_hyperparameters(self,cfg):
        
        self.encdec_optimizer = optim.Adam(self.encoderdecoder.parameters(),lr=cfg.lr)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        
        
    def load(self,encoder_path = 'Save/Meanpool_10.pt',decoder_path='Saved/SALSTM_10.pt'):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')

    def save(self,encoder_path,decoder_path):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            
            torch.save(model.encoder.state_dict(),encoder_path)
            torch.save(model.decoder.state_dict(),decoder_path)
        else:
            print('Invalid path address given.')
            
    def train_epoch(self,dataloader,utils):
        '''
        Function to train the model for a single epoch.
        Args:
         Input:
            dataloader : the dataloader object.basically train dataloader object.
         Return:
             epoch_loss : Average single time step loss for an epoch
        '''
        total_loss = 0
        start_iteration = 1
        print_loss = 0
        iteration = 1
        
        self.encoderdecoder.train()
        for data in dataloader:
            features, targets, mask, max_length, _,_,_ = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            loss = self.train_iter(utils,features,targets,mask,max_length,use_teacher_forcing)
            #print('Loss :',loss)
            print_loss += loss
            total_loss += loss
        # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".
                format(iteration, iteration / len(dataloader) * 100, print_loss_avg))
                print_loss = 0
             
            
            iteration += 1 
        return total_loss/len(dataloader)
        
        
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
        
        self.encdec_optimizer.zero_grad()
        
        loss = 0
        print_losses = []
        n_totals = 0
        
        input_variable = input_variable.permute(1,0,2).to(self.device)
        
        input_variable = input_variable.to(self.device)  
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        output = self.encoderdecoder(input_variable,target_variable,use_teacher_forcing) 
        #print(output.size())
        
        # Loss calculation
        for t in range(max_target_len):
            mask_loss, nTotal = utils.maskNLLLoss(output[t], target_variable[t], mask[t],self.device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
 
        # Perform backpropatation
        loss.backward()

        _ = nn.utils.clip_grad_norm_(self.encoderdecoder.parameters(), self.clip)
        self.encdec_optimizer.step()
        
        return sum(print_losses) / n_totals
    
    
    @torch.no_grad()
    def GreedyDecoding(self,features,max_length=15):
        batch_size = features.size()[0]
        features = features.permute(1,0,2).to(self.device)
        
        caption = self.encoderdecoder(features,use_teacher_forcing=False,training=False).permute(1,0) #(T,B,V)
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
        return caption,caps_text,
    
