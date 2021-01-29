import sys
import os
import re
import random
import unicodedata
import numpy as np
import torch
sys.path.append('pycocoevalcap/')
sys.path.append('pycocoevalcap/bleu')
sys.path.append('pycocoevalcap/cider')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# For Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#Convert target tensor to Caption
def target_tensor_to_caption(voc,target):
    gnd_trh = []
    lend = target.size()[1]
    for i in range(lend):
        tmp = ' '.join(voc.index2word[x.item()] for x in targets[:,i])
        gnd_trh.append(tmp)
    return gnd_trh

#Masked cross-entropy loss calculation; refers: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
def maskNLLLoss(inp, target, mask, device):
    inp = inp.squeeze(0)
    #print(inp.size(),target.size())
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp.squeeze(0), 1, target.view(-1, 1)).squeeze(1).float())
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

# Function to extract frames 
def FrameCapture(video_path, video_name): # For MSVD Sample every 10th frame
    #video_path = video_path_dict[video_name]
    # Path to video file 
    video_path = video_path+video_name  #Changes
    vidObj = cv2.VideoCapture(video_path) 
    count = 0
    fail = 0
    # checks whether frames were extracted 
    success = 1
    frames = []
    while success: 
        # OpenCV Uses BGR Colormap
        success, image = vidObj.read() 
        try:
            RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if count%10 == 0:            #Sample 1 frame per 10 frames
                frames.append(RGBimage)
            count += 1
        except:
            fail += 1
    vidObj.release()
    if count > 80:
        frames = frames[:81]
    return np.stack(frames[:-1]),count-1, fail

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

