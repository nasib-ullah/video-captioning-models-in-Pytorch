
'''
Module :  MARN model
Authors:  Nasibullah (nasibullah104@gmail.com)
Details : Implementation of the paper Memory Attended Recurrent Network for Video captioning.
          This implementation differ from original paper in 2 aspects.
          (1) During calculation of visual context information for memory, the attentions weights are not considered. mean pooling has been done 
              over frame features. Its not clear in the paper how to use attention values without propagating signal through the decoder.
          (2) Didn't consider the auxiliary features
          
Notations : B : Batch_size, T : Frame dimension, F : dimension of pre-trained CNN extracted features,
            F' : projected feature dimension.

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
        Encoder module. Project the video appearance and motion features into a different space which will be 
        send to the decoder.
        Argumets:
          input_size : CNN extracted feature size. [resnet101,Resnext101 - 2048, Inceptionv4,Inceptionresnetv2 - 1536]
          output_size : Dimention of projected space.
        '''
        
        self.appearance_projection_layer = nn.Linear(cfg.appearance_input_size,cfg.appearance_projected_size)
        self.motion_projection_layer = nn.Linear(cfg.motion_input_size ,cfg.motion_projected_size)
        
    def forward(self,appearance_feat,motion_feat):
        '''
        Project 2D and 3D convnet extracted features.
        Args:
            appearance_feat : (B,T,F)
            motion_feat : (B,T,F)
        Return:
            appearance_out : (B,T,F')
            motion_out : (B,T,F')
        
        '''
        appearance_out = self.appearance_projection_layer(appearance_feat)
        motion_out = self.motion_projection_layer(motion_feat)
        
        return appearance_out, motion_out
    
    
class TemporalAttention(nn.Module):
    def __init__(self,cfg):
        super(TemporalAttention,self).__init__()
        '''
        Temporal Attention module. It depends on previous hidden memory in the decoder,
        feature at the source side 
        at(s) = align(ht,hs)
              = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
        where
        score(ht,hs) = ht.t * hs                         (dot)
                     = ht.t * Wa * hs                  (general)
                     = va.t * tanh(Wa[ht;hs])           (concat)  
        Here we have used concat formulae.
        Argumets:
          hidden_size : hidden memory size of decoder. (batch,hidden_size)
          feat_size : feature size of each frame at encoder side.
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
        shape of hidden : (hidden_size) (batch,hidden_size) #(100,512)
        shape of feats : (batch size,feat_size)  #(100,40,1536)
        '''
        Wh = self.decoder_projection(hidden)  
        Uv = self.encoder_projection(feats)   
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.final_projection(torch.tanh(Wh+Uv))
        weights = F.softmax(energies, dim=1)
        weighted_feats = feats *weights.expand_as(feats)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats,weights

    
    
class RecurrentDecoder(nn.Module):
    
    def __init__(self,cfg,voc):
        super(RecurrentDecoder, self).__init__()
        '''
        Attention-based Recurrent Decoder
        Args:
            cfg : Configuration object corresponding to MARN
            voc : Vocabulary object
        
        '''
        
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

    
    def forward(self, inputs, hidden, appearance_feats,motion_feats):
        '''
        we run this one step (word) at a time
        
        inputs -  (1, B)
        hidden - h_n/c_n :(num_layers * num_directions, B, hidden_size)    # GRU:h_n   LSTM:(h_n,c_n)
        appearance_feats - (B,T,F')
        motion_feats : (B,T,F')
        
        '''
        embedded = self.embedding(inputs) # [i/p:(1,batch)  o/p:(1,batch,embedding_size)]
        last_hidden = hidden[0] if self.decoder_type=='lstm' else hidden
        last_hidden = last_hidden.view(self.n_layers,last_hidden.size(1),last_hidden.size(2))
        last_hidden = last_hidden[-1]
        appearance_feats, appearance_weights = self.attention(last_hidden,appearance_feats) #(100,1536) #(100,28,1)
        motion_feats, motion_weights = self.attention(last_hidden,motion_feats) #(100,1536) #(100,28,1)
        context_vector = torch.cat((appearance_feats,motion_feats),dim=1).unsqueeze(0) (1,B,512*2)
        
        input_combined = torch.cat((embedded,context_vector),dim=2)
        output, hidden = self.rnn(input_combined, hidden) # (1,100,512)
        output = output.squeeze(0) # (100,512)
        output = self.out(output) # (100,num_words)
        output = F.softmax(output, dim = 1) #(100,num_words)
        return output, hidden, appearance_weights 
    

    
class MARN(nn.Module):
    
    def __init__(self,voc,cfg,path):
        super(MARN,self).__init__()

        self.voc = voc
        self.path = path
        self.cfg = cfg
        self.word_memory = {}
        
        
        self.encoder = Encoder(cfg).to(cfg.device)
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)

        self.decoder = RecurrentDecoder(cfg,voc).to(cfg.device)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr,amsgrad=True)
    
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
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
        
        
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)
        
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr,amsgrad=True)
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
        self.encoder.train()
        self.decoder.train()
        for data in dataloader:
            features, targets, mask, max_length, _,_,_ = data #change
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            loss = self.train_iter(utils,appearance_features,motion_features,
                                   targets,mask,max_length,use_teacher_forcing)
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
        
        
    def train_iter(self,utils,input_variable,motion_variable,target_variable,
                   mask,max_target_len,use_teacher_forcing):
        '''
        Forward propagate input signal and update model for a single iteration. 
        
        Args:
        Inputs:
            input_variable : video mini-batch tensor; size = (B,T,F)
            motion_variable : video motion tensor; size = (B,T,F)
            target_variable : Ground Truth Captions;  size = (T,B); T will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T,C)
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
        motion_variable = motion_variable.to(self.device)
        
        input_variable,motion_variable = self.encoder(input_variable,motion_variable)  
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
                decoder_output, decoder_hidden,attn_weight = self.decoder(decoder_input, decoder_hidden,
                                                input_variable.float(),motion_variable.float())
                # Teacher forcing: next input comes from ground truth(data distribution)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t],self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,attn_weight = self.decoder(decoder_input,
                                            decoder_hidden,input_variable.float(),motion_variable.float())
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

        
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
            
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        
        
        return sum(print_losses) / n_totals
    
    def _calculate_AC_loss(self,alphas):
        '''
        Calculate Attention-Coherent Loss.
        '''
        alphas = alphas.squeeze(2)
        alpha_next = alphas[:,1:]
        alpha_previous = alphas[:,:-1]
        ac_loss = torch.abs(alpha_next - alpha_previous).sum()
        
        return ac_loss
    
    def _generate_word2videos(self,data_handler):
        vid2words = {}
        self.word2vid = {}
        word_list = list(self.voc.word2index.keys())[3:]
        for k,v in data_handler.train_dict.items():
            words = ' '.join([x for x in v]).split(' ')
            vid2words[k] = words
        for word in word_list:
            temp = []
            for k,v in vid2words.items():
                if word in v:
                    temp.append(k)
            self.word2vid[word] = temp 
        self.word_list = word_list
            
    def _generate_visual_context_vector(self,word,data_handler):
        appearance_list = []
        motion_list = []
        vid_list = self.word2vid[word]
        for vid in vid_list:
            apr,mtn = self.encoder(torch.tensor(data_handler.appearance_feature_dict[vid]).to(self.device),
                         torch.tensor(data_handler.appearance_feature_dict[vid]).to(self.device))
            appearance_list.append(apr.mean(dim=0))
            motion_list.append(mtn.mean(dim=0))
        if len(vid_list) == 0:
            appearance_list.append(torch.zeros(self.cfg.appearance_projected_size))
            motion_list.append(torch.zeros(self.cfg.motion_projected_size))
        appearance_tensor = torch.stack(appearance_list).mean(dim=0)
        motion_tensor = torch.stack(motion_list).mean(dim=0)
        
        gr = appearance_tensor + motion_tensor
        return gr.detach()
    
    def _generate_word_embedding(self,word):
        word = torch.tensor([[self.voc.word2index[word]]])
        word_feat = self.decoder.embedding(word.to(self.device)).detach()
        return word_feat
        
    def _generate_auxiliary_features(self):
        pass

    def generate_memory(self,data_handler):
        self._generate_word2videos(data_handler)
        for word in self.word_list:
            er = self._generate_word_embedding(word)
            gr = self._generate_visual_context_vector(word,data_handler)
            #auxiliary_feat = self._generate_auxiliary_features(word)
            self.word_memory[word] = (gr,er)
        
        
    @torch.no_grad()
    def GreedyDecoding(self,features,motion_features,max_length=15):
        batch_size = features.size()[0]
        features = features.to(self.device)
        motion_features = motion_features.to(self.device)
        
        
        features,motion_features = self.encoder(features,motion_features)  
        
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(batch_size)]]).to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, batch_size,
                                      self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        caption = []
        attention_values = []
        for _ in range(max_length):
            decoder_output, decoder_hidden,attn_values = self.decoder(decoder_input, 
                                                    decoder_hidden,features.float(),motion_features.float())
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
    def BeamDecoding(self,feats, motion_feats,width, alpha=0.,max_caption_len = 15):
        batch_size = feats.size(0)
        vocab_size = self.voc.num_words
        
        feats,motion_feats = self.encoder(feats,motion_feats)  
        

        vfunc = np.vectorize(lambda t: self.voc.index2word[t]) # to transform tensors to words
        rfunc = np.vectorize(lambda t: '' if t == 'EOS' else t) # to transform EOS to null string
        lfunc = np.vectorize(lambda t: '' if t == 'SOS' else t) # to transform SOS to null string
        pfunc = np.vectorize(lambda t: '' if t == 'PAD' else t) # to transform PAD to null string

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
                output, next_hidden, _ = self.decoder(input, hidden, feats,motion_feats) # need to check

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

    
