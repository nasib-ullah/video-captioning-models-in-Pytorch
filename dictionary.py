'''
Module :  dictionary
Author:  Nasibullah (nasibullah104@gmail.com)
          
'''

import pickle
import os


class Vocabulary:
    
    def __init__(self, cfg):
        self.name = cfg.dataset
        self.cfg = cfg
        self.trimmed = False
        self.word2index = {"PAD":cfg.PAD_token,"EOS":cfg.EOS_token,"SOS":cfg.SOS_token, "UNK":cfg.UNK_token}
        self.word2count = {}
        self.index2word = {cfg.PAD_token:"PAD",cfg.EOS_token:"EOS",cfg.SOS_token:"SOS", cfg.UNK_token:"UNK"}
        self.num_words = 4
        
    def addSentence(self,sentence): #Add Sentence to vocabulary
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self, word):  # Add words to vocabulary
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            if self.trimmed == False:
                self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            if self.trimmed == False:
                self.word2count[word] += 1
            
    def save(self,word2index_dic = 'word2index_dic.p', index2word_dic = 'index2word_dic.p',
         word2count_dic = 'word2count_dic.p'):

        w2i = os.path.join('Saved',self.name+'_'+word2index_dic)
        i2w = os.path.join('Saved',self.name+'_'+index2word_dic)
        w2c = os.path.join('Saved',self.name+'_'+word2count_dic)
        try:
            with open(w2i, 'wb') as fp:
                pickle.dump(self.word2index, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open(i2w, 'wb') as fp:
                pickle.dump(self.index2word, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open(w2c, 'wb') as fp:
                pickle.dump(self.word2count, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print('Path Error, Verify the path of the filename is correct')



    def load(self, word2index_dic = 'word2index_dic.p', index2word_dic = 'index2word_dic.p',
             word2count_dic = 'word2count_dic.p'):

        w2i = os.path.join('Saved',self.name+'_'+word2index_dic)
        i2w = os.path.join('Saved',self.name+'_'+index2word_dic)
        w2c = os.path.join('Saved',self.name+'_'+word2count_dic)

        try:        
            with open(w2i, 'rb') as fp:
                self.word2index = pickle.load(fp)
            
            with open(i2w, 'rb') as fp:
                self.index2word = pickle.load(fp)
            
            with open(w2c, 'rb') as fp:
                self.word2count = pickle.load(fp)
            
            self.num_words = len(self.word2index)

        except:
            print('File loading error.. check the path or filename is correct')

        
    def trim(self, min_count):  # Trim Rare words with frequency less than min_count
        if self.trimmed:
            print('Already trimmed before')
            return 0
        self.trimmed = True
        
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD":self.cfg.PAD_token,"EOS":self.cfg.EOS_token,"SOS":self.cfg.SOS_token, "UNK":self.cfg.UNK_token}
        #self.word2count = {}
        self.index2word = {self.cfg.PAD_token:"PAD",self.cfg.EOS_token:"EOS",self.cfg.SOS_token:"SOS", self.cfg.UNK_token:"UNK"}
        self.num_words = 4

        for word in keep_words:
            self.addWord(word)
            if word not in self.word2count:
                del self.word2count[word]

