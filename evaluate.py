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
    
    def __init__(self,model,dataloader,path,cfg,reference_dict):
        
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

    def prediction_list(self,model):
        self.prediction_dict = {}
        ide_list = []
        caption_list = []
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                features, targets, mask, max_length,ides= data
                if self.cfg.model_name == 'mean_pooling':
                    cap,cap_txt = model.GreedyDecoding(features.to(self.cfg.device))
                if self.cfg.model_name == 'sa-lstm':
                    cap,cap_txt,_ = model.GreedyDecoding(features)
                ide_list += ides
                caption_list += cap_txt
        for a in zip(ide_list,caption_list):
            self.prediction_dict[str(a[0])] = [a[1].strip()]
            
    def evaluate(self,scorer,model,epoch,loss=9999):
        self.prediction_list(model)
        scores = scorer.score(self.reference_dict,self.prediction_dict)
        self.scores[epoch] = scores
        self.losses[epoch] = loss
        if scores['METEOR']>self.meteor:
            self.meteor = scores['METEOR']
            self.best_model = model
        if scores['METEOR']>self.meteor_sota:
            self.save_model(model,epoch)
        return scores
        

    def save_model(self,model,epoch):
        print('Better result saving models....')
        filename = os.path.join(self.cfg.saved_models_path, self.cfg.model_name+str(epoch)+'.pt')
        torch.save(model,filename)
