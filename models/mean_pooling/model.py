'''
Module :  Mean Pooling model
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
        self.layer = nn.Linear(cfg.input_size,cfg.videofeat_size)
        
    def forward(self,x):
        out = self.layer(x)
        return out
    
    
class DecoderRNN(nn.Module):
    def __init__(self,cfg,voc):
        super(DecoderRNN, self).__init__()
        '''
        Decoder, Basically a language model.
        
        Arguments:
        hidden_size : hidden memory size of LSTM/GRU
        output_size : output size. Its same as the vocabulary size.
        n_layers : 
        '''
        # Keep for reference
        self.hidden_size = cfg.hidden_size
        self.output_size = voc.num_words
        self.n_layers = cfg.n_layers
        self.dropout = cfg.dropout
        self.decoder_type = cfg.decoder_type
        # Define layers
        self.embedding = nn.Embedding(voc.num_words, cfg.hidden_size)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        if self.decoder_type == 'gru':
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout))
        else:
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size,
                           self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout))
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_step, last_hidden):
        '''
        we run this one step (word) at a time
        
        inputs -  (1, batch)
        hidden - (num_layers * num_directions, batch, hidden_size)
        feats - (batch,attention_length,annotation_vector_size) 
        
        '''
        #last_hidden = hidden[0] if isinstance(hidden,tuple) else hidden
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.rnn(embedded, last_hidden)
        output = self.out(rnn_output)
        output = F.softmax(output, dim = 2)
        # Return output and final hidden state
        return output, hidden
    
class MeanPooling(nn.Module):
    
    def __init__(self,voc,cfg,path):
        super(MeanPooling,self).__init__()
        self.voc = voc
        self.path = path
        self.cfg = cfg
        self.encoder = Encoder(cfg).to(cfg.device)
        self.decoder = DecoderRNN(cfg,voc).to(cfg.device)
        
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.print_every = cfg.print_every
        self.clip = cfg.clip
        self.device = cfg.device
        
    def update_hyperparameters(self,cfg):
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        
    def load(self,encoder_path = 'Save/Meanpool_10.pt',decoder_path='Save/Meanpool_10.pt'):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')

    def save(self,encoder_path, decoder_path):
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
        self.encoder.train()
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
            input_variable : mini-batch tensor; size = (B,F); B - batch_size, F - feature dimension
            target_variable : Ground Truth Captions;  size = (T,B); T - maximum seuence length for the mini batch, will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T,B)
            max_target_len : maximum lengh of the mini-batch; size = T
            use_teacher_forcing : binary variable. If True training uses teacher forcing else sampling.
            clip : clip the gradients to counter exploding gradient problem.
        Returns:
            iteration_loss : average loss per time step.
        '''
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        
        loss = 0
        print_losses = []
        n_totals = 0
        
        input_variable = input_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        # Forward pass through encoder
        encoder_output = self.encoder(input_variable).unsqueeze_(0)
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(self.cfg.batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = torch.cat(self.cfg.n_layers*[encoder_output]) #notice
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # Teacher forcing: next input comes from ground truth(data distribution)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t],self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
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
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.enc_optimizer.step()
        self.dec_optimizer.step()

        return sum(print_losses) / n_totals
    
    @torch.no_grad()
    def GreedyDecoding(self,features,max_length=15):
        encoder_output = self.encoder(features).unsqueeze_(0)
        batch_size = features.size()[0]
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(batch_size)]]).to(self.device)
        #print(decoder_input.size())
        decoder_hidden = encoder_output 
        decoder_hidden = torch.cat(self.cfg.n_layers*[encoder_output])
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        caption = []
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.squeeze(0).topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(self.device)
            caption.append(topi.squeeze(1).cpu())
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
        return caption,caps_text
        
    @torch.no_grad()
    def BeamDecoding(self,features,beam_length=3, max_length=15,return_single=True):
        '''
        Beam decoding for Mean Pooling
        Args:
            Input :
                features : Encoder output feature
                beam_length : Beam length
                max_length : maximum sentence length to generate
                return_single : to select whether to return top-1 result or top-beam_length results
             
            Return:
                final_captions : output tensor  # shape - (beam_length,batch_size,max_length) 
                caps_text : text after EOS removed 
                final_scores : Beam score # shape - (batch_size,beam_length)
        
        '''
        batch_size = features.size()[0]
        vfunc = np.vectorize(lambda t: self.voc.index2word[t]) # to transform tensors to words
        rfunc = np.vectorize(lambda t: '' if t == 'EOS' else t) # to transform EOS to null string
        
        final_captions = torch.zeros(beam_length,batch_size,max_length)
        final_scores = torch.tensor([[0]*beam_length for i in range(batch_size)])
        encoder_output = self.encoder(features).unsqueeze_(0)
        
        decoder_input = final_captions[:,:,0].unsqueeze_(1).long().to(self.device)
        decoder_hidden = encoder_output 
        decoder_hidden = torch.cat(self.cfg.n_layers*[encoder_output])
        decoder_hidden = torch.stack([decoder_hidden]*beam_length).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        for l in range(max_length):
            beam_output = []
            for i in range(beam_length):
                #split data along the beam dimension and concatenate results
                if self.cfg.decoder_type == 'lstm':
                    decoder_output, (decoder_hidden[0][i],decoder_hidden[1][i]) = self.decoder(decoder_input[i],
                                                                 (decoder_hidden[0][i],decoder_hidden[1][i]))
                else:
                    decoder_output,decoder_hidden[i] = self.decoder(decoder_input[i],decoder_hidden[i])
                    
                # add log prob 
                tmp = (-1)*np.log(decoder_output.squeeze(0).cpu().numpy()) + final_scores[:,i].view(batch_size,1).cpu().numpy()
                beam_output.append(tmp)
                
            beam_output = np.concatenate(beam_output,1) 
            value,index = torch.tensor(beam_output).topk(beam_length)
            #Add this to final captions and final scores
            final_scores = value
            for i,ind in enumerate(index.permute(1,0),0): # need to loop over batches
                for b in range(len(ind)):  
                    kk = int(ind[b].item()/self.voc.num_words)
                    final_captions[i,b,:(l+1)] = torch.cat([final_captions[kk,b,:l].view(-1),ind[b].float().view(-1)])
               
                        
            #set decoder input
            index = index % self.voc.num_words
            decoder_input = index.unsqueeze_(1).permute(2,1,0).to(self.device)
        final_captions = final_captions % self.voc.num_words
        caps_text = []
        captions = vfunc(final_captions.cpu().numpy())
        captions = rfunc(captions)
        if return_single:
            for eee in captions:
                caps_text.append(' '.join(x for x in eee[2]))
        else:
            for eee in captions:
                for jj in eee:
                    caps_text.append(' '.join(x for x in jj))
        
        return final_captions,caps_text,final_scores
            
