'''
Module :  SA-LSTM model
Authors:  Nasibullah (nasibullah104@gmail.com)
Beam decoding will be added in future
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
          hidden_size : hidden memory size of decoder.
          feat_size : feature size of each grid (annotation vector) at encoder side.
          bottleneck_size : intermediate size.
        '''
        self.hidden_size = cfg.hidden_size
        self.feat_size = cfg.feat_size
        self.bottleneck_size = cfg.attn_size
        
        self.decoder_projection = nn.Linear(self.hidden_size,self.bottleneck_size)
        self.encoder_projection = nn.Linear(self.feat_size, self.bottleneck_size)
        self.final_projection = nn.Linear(self.bottleneck_size,1)
     
    def forward(self,hidden,feats):
        '''
        shape of hidden (hidden_size)
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
        self.feat_size = cfg.feat_size
        self.feat_len = cfg.feat_len
        self.embedding_size = cfg.embedding_size
        self.hidden_size = cfg.hidden_size
        self.attn_size = cfg.attn_size
        self.output_size = voc.num_words
        self.rnn_dropout = cfg.rnn_dropout
        self.n_layers = cfg.n_layers
        self.decoder_type = cfg.decoder_type

        # Define layers
        self.embedding = nn.Embedding(voc.num_words, self.embedding_size)
        self.attention = TemporalAttention(cfg)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        if self.decoder_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size+self.feat_size, self.hidden_size,
                              self.n_layers, dropout=(0 if self.n_layers == 1 else self.rnn_dropout))
        else:
            self.rnn = nn.LSTM(self.embedding_size+self.feat_size, self.hidden_size,
                           self.n_layers, dropout=(0 if self.n_layers == 1 else self.rnn_dropout))
        self.out = nn.Linear(self.hidden_size, self.output_size)

    
    def forward(self, inputs, hidden, feats):
        '''
        we run this one step (word) at a time
        
        inputs -  (1, batch)
        hidden - (num_layers * num_directions, batch, hidden_size)
        feats - (batch,attention_length,annotation_vector_size) 
        
        '''
        embedded = self.embedding(inputs)
        if self.n_layers > 1:
            if self.decoder_type == 'lstm':
                last_hidden = hidden[0][0]
            else:
                last_hidden = hidden[0]
        else:
            last_hidden = hidden[0]
        feats, attn_weights = self.attention(last_hidden.squeeze(0),feats)
        input_combined = torch.cat((embedded,feats.unsqueeze(0)),dim=2)
        output, hidden = self.rnn(input_combined, hidden)
        output = output.squeeze(0)
        output = self.out(output)
        output = F.softmax(output, dim = 1)
        return output, hidden, attn_weights
    
    
    
    
class SALSTM(nn.Module):
    
    def __init__(self,voc,cfg,path):
        super(SALSTM,self).__init__()

        self.voc = voc
        self.path = path
        self.cfg = cfg
        self.decoder = DecoderRNN(cfg,voc).to(cfg.device)
        
        self.optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.print_every = cfg.print_every
        self.clip = cfg.clip
        self.device = cfg.device
        
    def update_hyperparameters(self,cfg):
        self.optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        
        
    def load(self,path='Saved/SALSTM_10.pt'):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')

    def save(self,decoder_path):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
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
        self.decoder.train()
        for data in dataloader:
            features, targets, mask, max_length,_ = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            loss = self.train_iter(utils,features,targets,mask,max_length,use_teacher_forcing)
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
            input_variable : image mini-batch tensor; size = (B,C,W,H)
            target_variable : Ground Truth Captions;  size = (T,B); T will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T,C)
            max_target_len : maximum lengh of the mini-batch; size = T
            use_teacher_forcing : binary variable. If True training uses teacher forcing else sampling.
            clip : clip the gradients to counter exploding gradient problem.
        Returns:
            iteration_loss : average loss per time step.
        '''
        self.optimizer.zero_grad()
        
        loss = 0
        print_losses = []
        n_totals = 0
        
        input_variable = input_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        # Forward pass through encoder
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(self.cfg.batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, self.cfg.batch_size,
                                      self.cfg.hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,_ = self.decoder(decoder_input, decoder_hidden,input_variable.float())
                # Teacher forcing: next input comes from ground truth(data distribution)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t],self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,_ = self.decoder(decoder_input, decoder_hidden,input_variable.float())
                # No teacher forcing: next input is decoder's own current output(model distribution)
                _, topi = decoder_output.squeeze(0).topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.cfg.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output, target_variable[t], mask[t],self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.optimizer.step()
        
        return sum(print_losses) / n_totals
    
    
    @torch.no_grad()
    def GreedyDecoding(self,features,max_length=15):
        features = features.to(self.device)
        batch_size = features.size()[0]
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(batch_size)]]).to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, batch_size,
                                      self.cfg.hidden_size).to(self.device)
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

