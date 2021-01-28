import os
import random
import itertools
from PIL import Image
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F


from dictionary import Vocabulary,EOS_token,PAD_token,SOS_token,UNK_token
from utils import maskNLLLoss,normalizeString

def collate_fn(batch):
    data=[item[0] for item in batch]
    images=torch.stack(data,0)
    
    ides = [item[2] for item in batch]
    #ides = torch.stack(ides_list,0)
    
    label=[item[1] for item in batch]
    max_target_len = max([len(indexes) for indexes in label])
    padList = list(itertools.zip_longest(*label, fillvalue = 0))
    
    lengths = torch.tensor([len(p) for p in label])
    padVar = torch.LongTensor(padList)
    
    m = []
    for i, seq in enumerate(padVar):
        #m.append([])
        tmp = []
        for token in seq:
            if token == 0:
                tmp.append(int(0))
            else:
                tmp.append(1)
        m.append(tmp)
    m = torch.tensor(m)

    return images, padVar, m, max_target_len, ides

class MSVD(Dataset):
    
    def __init__(self, feature_dict, annotation_dict , video_name_list, voc, vid2url):
        
        self.annotation_dict = annotation_dict
        self.feature_dict = feature_dict
        self.v_name_list = video_name_list
        self.voc = voc
        self.vid2url = vid2url
        
    def __len__(self):
        return len(self.v_name_list)
    
    def __getitem__(self,idx):
        
        anno = random.choice(self.annotation_dict[self.v_name_list[idx]])
        anno_index = [self.voc.word2index[word] for word in anno.split(' ')] + [EOS_token]
        return torch.tensor(self.feature_dict[self.v_name_list[idx]]), anno_index, self.v_name_list[idx]


class DataHandler:
    
    def __init__(self,cfg,path,voc):
        
        self.vid2url = self._name_mapping(path)
        
        self.train_dict = self._file_to_dict(path.train_annotation_file)
        self.val_dict = self._file_to_dict(path.val_annotation_file)
        self.test_dict = self._file_to_dict(path.test_annotation_file)

        self.train_name_list = list(self.train_dict.keys())
        self.val_name_list = list(self.val_dict.keys())
        self.test_name_list = list(self.test_dict.keys())
        
        self.feature_dict = self._feature_to_dict(path)
        self.voc = voc
        self.cfg = cfg
        
    def _file_to_dict(self,path):
        dic = dict()
        fil = open(path,'r+')
        for f in fil.readlines():
            l = f.split() 
            ll = ' '.join(x for x in l[1:])
            if l[0] not in dic:
                dic[l[0]] = [ll]
            else:
                dic[l[0]].append(ll)
        return dic
    
    def _name_mapping(self,path):
        vid2url = dict()
        fil = open(path.name_mapping_file,'r+')
        for f in fil.readlines():
            l = f.split(' ')
            vid2url[l[1].strip('\n')] = l[0]
        return vid2url
    
    #Loading features
    def _feature_to_dict(self,path):
        feature_dict = dict()
        v_name_list = list(self.vid2url.keys())
        for name in v_name_list:
            tmp = os.path.join(path.feature_path,name)
            arr = np.load(tmp+'.npy')
            feature_dict[name] = np.mean(arr,axis=0)        
        return feature_dict
    
    def getDatasets(self):
        train_dset = MSVD(self.feature_dict, self.train_dict, self.train_name_list, self.voc, self.vid2url)
        val_dset = MSVD(self.feature_dict, self.val_dict, self.val_name_list, self.voc, self.vid2url)
        test_dset = MSVD(self.feature_dict, self.test_dict, self.test_name_list, self.voc, self.vid2url)
        return train_dset, val_dset, test_dset
    
    def getDataloader(self,train_dset,val_dset,test_dset):
        
        train_loader=DataLoader(train_dset,batch_size = self.cfg.batch_size, num_workers = 8,shuffle = True,
                        collate_fn = collate_fn, drop_last=True)

        val_loader = DataLoader(val_dset,batch_size = 10, num_workers = 8,shuffle = False,
                         drop_last=False)
        test_loader = DataLoader(test_dset,batch_size = 10, num_workers = 8,shuffle = False,
                         drop_last=False)
        
        return train_loader,val_loader,test_loader

