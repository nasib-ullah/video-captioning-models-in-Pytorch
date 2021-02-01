import os
import sys
import torch
import json
import pickle
from utils import score

class Evaluator:
    
    def __init__(self,model_name,prediction_filepath,dataloader,cfg,reference_dict):
        
        self.arch_name = model_name
        self.cfg = cfg
        self.prediction_filepath = prediction_filepath
        self.dataloader = dataloader
        self.reference_dict = reference_dict
        self.prediction_dict = {}
        self.scores = {}
        self.meteor = 0.296 # save best model based on METEOR score

    def prediction_list(self,model):
        self.prediction_dict = {}
        ide_list = []
        caption_list = []
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                features, targets, mask, max_length,ides= data
                cap,cap_txt = model.GreedyDecoding(features.to(self.cfg.device))
                ide_list += ides
                caption_list += cap_txt
        for a in zip(ide_list,caption_list):
            self.prediction_dict[str(a[0])] = [a[1].strip()]
            
    def evaluate(self,model,epoch):
        self.prediction_list(model)
        scores = score(self.reference_dict,self.prediction_dict)
        self.scores[epoch] = scores
        if scores['METEOR']>self.meteor:
            self.meteor = scores['METEOR'] 
            self.save_model(model,epoch)
        return scores
        

    def save_model(self,model,epoch):
        print('Better result saving models....')
        encoder_filename = 'Save/'+ self.arch_name+'encoder_'+str(epoch)+'.pt'
        decoder_filename = 'Save/'+ self.arch_name+'decoder_'+str(epoch)+'.pt'
        torch.save(model.encoder.state_dict(),encoder_filename)
        torch.save(model.decoder.state_dict(),decoder_filename)
        
