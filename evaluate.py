'''
Module : evaluate
Author : Nasibullah (nasibullah104@gmail.com)
'''

import os
import sys
import torch
import json
import pickle

class Evaluator:
    
    def __init__(self,model,dataloader,path,cfg,reference_dict,decoding_type = 'greedy'):
        '''
        Decoding type : {'greedy','beam','beam_beta}
        '''
        self.path = path
        self.cfg = cfg
        self.dataloader = dataloader
        self.reference_dict = reference_dict
        self.prediction_dict = {}
        self.scores = {}
        self.meteor = 0.32 # save best model based on METEOR score
        self.losses = {}
        self.best_model = model
        self.meteor_sota = 0.34
        self.decoding_type = decoding_type

    def prediction_list(self,model):
        self.prediction_dict = {}
        ide_list = []
        caption_list = []
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                features, targets, mask, max_length,ides,motion_feat,object_feat= data
                if self.cfg.model_name == 'mean_pooling':
                    if self.decoding_type == 'greedy':
                        cap,cap_txt = model.GreedyDecoding(features.to(self.cfg.device))
                    if self.decoding_type == 'beam':
                        tensor,cap_txt,scores = model.BeamDecoding(features.to(self.cfg.device),return_single=True)
                    
                if self.cfg.model_name == 'sa-lstm':
                    if self.decoding_type == 'greedy':
                        cap,cap_txt,_ = model.GreedyDecoding(features.to(self.cfg.device))
                    if self.decoding_type == 'beam':
                        cap_txt = model.BeamDecoding(features.to(self.cfg.device),self.cfg.beam_length)
                        
                        
                if self.cfg.model_name == 'recnet':
                    if self.decoding_type == 'greedy':
                        cap,cap_txt,_ = model.GreedyDecoding(features.to(self.cfg.device))
                    else:
                        cap_txt = model.BeamDecoding(features.to(self.cfg.device),self.cfg.beam_length)
                    
                if self.cfg.model_name == 's2vt':
                    if self.decoding_type == 'greedy':
                        pass #yet to implement
                    else:
                        pass # yet ti implement
                ide_list += ides
                caption_list += cap_txt
        for a in zip(ide_list,caption_list):
            self.prediction_dict[str(a[0])] = [a[1].strip()]
            
    def evaluate(self,scorer,model,epoch,loss=9999):
        self.prediction_list(model)
        scores = scorer.score(self.reference_dict,self.prediction_dict)
        self.scores[epoch] = scores
        self.losses[epoch] = loss
        return scores



    def save_model(self,model,epoch):
        print('Saving models....')
        filename = os.path.join(self.path.saved_models_path, self.cfg.model_name+str(epoch)+'.pt')
        torch.save(model,filename)
