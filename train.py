import os, sys
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
#from torch.nn import init
from torch.utils.data import Subset, ConcatDataset, TensorDataset


def _longtail_train_indices(full_dataset, num_classes, imb_factor, imb_seed=None):
    """Return indices for long-tail (exponential) training subset. Class k has n_max * (imb_factor)^(-k/(num_classes-1)) samples."""
    if imb_factor is None or imb_factor <= 1:
        return None
    targets = np.asarray(getattr(full_dataset, "targets", getattr(full_dataset, "labels", None))).ravel()
    if targets is None or len(targets) == 0:
        return None
    rng = np.random.default_rng(imb_seed)
    max_per_class = len(targets) // num_classes
    indices = []
    for k in range(num_classes):
        class_idx = np.where(targets == k)[0]
        n_k = int(max_per_class * (imb_factor ** (-k / max(1, num_classes - 1))))
        n_k = max(1, min(n_k, len(class_idx)))
        chosen = rng.choice(class_idx, size=n_k, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return indices

#models
#from torchvision.models import resnet18
#from torchvision.models import resnet50
from torchvision.datasets.vision import VisionDataset
import torchvision.models as tv_models
#import torchvision.models as models_

#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
#import torch.multiprocessing as mp

import argparse, copy
import pickle, time, random
from PIL import Image
import numpy as np

# DataLoader reproducibility: workers read seed from env (set in main() before creating dataset)
def _seed_worker(worker_id):
    """Seed each DataLoader worker so transforms (e.g. RandomHorizontalFlip) are reproducible across runs."""
    seed_val = os.environ.get("TRAIN_DATALOADER_SEED", "")
    if not seed_val:
        return
    try:
        base = int(seed_val)
        np.random.seed(base + worker_id)
        random.seed(base + worker_id)
    except ValueError:
        pass
import wandb
import math
from collections import deque

#from math import log10, sqrt, log2, log

from models import *

#from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration
#from netcal.binning import IsotonicRegression
from ensemble_fusion import FusionClassifier, AdversarialTrainingClassifier
from cals import AugLagrangian, AugLagrangianClass

from ecg_loss import ecg_loss
from robust_losses import focal_loss, clue_lite_loss, trades_loss, mart_loss, pgd_attack_ce, clue_loss
LOSS_ECG = "ecg"
LOSS_FOCAL = "focal"
LOSS_CLUE_LITE = "clue_lite"
LOSS_CLUE = "clue"
_AUTO_LAM_RULES = ("auto", "auto_w", "auto_d", "auto_dw", "auto_tr", "auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate")

# --- PE mode switches ---
PE_MODE = "raw"          # "raw" | "logk" | "logk_rms" | "none" (none/raw = no PE norm)
PE_RMS_BETA = 0.99       # EMA beta
PE_RMS_EPS = 1e-8

Normalize_entropy = (PE_MODE in ("logk", "logk_rms"))
USE_PE_RMS = (PE_MODE == "logk_rms")

imageNet_original = os.environ.get("IMAGENET_ORIGINAL", "0").lower() in ("1","true","yes","y")
LOSS_MIN_CROSSENT = 0 # minimize cross entropy loss
LOSS_MIN_CROSSENT_UNC  = 1 # minimize cross_entropy_loss + uncertainty
LOSS_MIN_CROSSENT_MAX_UNC = 2 # minimize cross_entropy_loss - uncertainty = minimize cross_entropy_loss + maximize uncertainty 
LOSS_MIN_UNC = 3 # minimize uncertainty
LOSS_MAX_UNC = 4 # minimize -uncertainty = maximize uncertainty 
LOSS_MIN_BINARYCROSSENT = 5 # minimize binary  cross entropy loss

UNCERTAINTY_MEASURE = "PE" 
#UNCERTAINTY_MEASURE = "MI" 


#LOSS_2nd_stage = LOSS_MAX_UNC
#LOSS_2nd_stage_wrong = LOSS_MIN_CROSSENT_MAX_UNC # before
#LOSS_2nd_stage_correct = LOSS_MIN_CROSSENT
#LOSS_2nd_stage_correct = LOSS_MIN_CROSSENT_UNC # before

LOSS_2nd_stage_wrong = LOSS_ECG
LOSS_2nd_stage_correct = LOSS_ECG

#option_stage2 = 'mix'
#option_stage2 = 'batch_mix'
option_stage2 = 'batch_mix2'
#option_stage2 = 'batch_granularity'

#Normalize_entropy =  False #   True # 
cycle_lr = False #False #

normalization = False #  True #
normalization_sum = False #   True #   
standardization = False # True # 

Weighted_Sum = False # True # 
Adaptive_Balancing =  False #  True #
dynamic_weights =    False # True #

#for calibration - used only in binary classification problems
PlattScaling_Flag =  False 
IsotonicRegression_Flag = False #True  
TemperatureScaling_Flag = False
BetaCalibration_Flag =  False

print("MEASURE OF UNCERTAINTY IS " + UNCERTAINTY_MEASURE + ' LOSS 2ND STAGE IS ' + str(LOSS_2nd_stage_correct) + ' & ' + str(LOSS_2nd_stage_wrong) + ' OPTION 2ND STAGE ' +  option_stage2)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def entropy(output):
    """Calculates the entropy loss of a probability distribution."""
    log_probs = F.log_softmax(output, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    # entropy of each prediction
    return entropy

class Uncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe_rms_v = None  # EMA of E[p^2]

    def entropy_loss(self, model, X, y, num_samples=10, reduction='mean'):
        """Calculates the entropy loss of a probability distribution."""
        probs = None 
        for _ in range(num_samples):
            #MC dropout
            if hasattr(model, "enable_dropout"):
                model.enable_dropout()
            softmax_output = F.softmax( model(X), dim=1)

            if probs is None: probs = torch.zeros_like(softmax_output)
            probs = probs+softmax_output

            del softmax_output
            torch.cuda.empty_cache()

        probs /= float(num_samples)
        mask = probs == 0  # Create a mask of zero values
        probs[mask] = 10e-20  # Replace zero values with small_value

        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        if Normalize_entropy:
            _num_clases = torch.Tensor([probs.size(dim=1)]).to(entropy.device) 
            entropy_max = torch.log(_num_clases)
            entropy = torch.div(entropy, entropy_max) # normalized predictive entropy

        # --- NEW: Adam-style RMS stabilization on PE_norm ---
        if USE_PE_RMS:
            # batch second moment (scalar), no gradient through statistics
            m_t = torch.mean(entropy.detach() ** 2)

            if self.pe_rms_v is None:
                self.pe_rms_v = m_t
            else:
                self.pe_rms_v = PE_RMS_BETA * self.pe_rms_v + (1.0 - PE_RMS_BETA) * m_t

            entropy = entropy / (torch.sqrt(self.pe_rms_v) + PE_RMS_EPS)

        if reduction=='mean':
            loss = torch.mean(entropy) #mean entropy
        elif reduction=='sum':
            loss = torch.sum(entropy) #total entropy
        else:
            loss = entropy

        #probs = np.mean(outputs,axis=0) #p(y|D,x) = 1/T sum_T p(y|w_i,x)
        #probs = self._check_logs_zero(probs)
        #entropy_vals = -np.sum(probs * log_probs, axis=1)# predictive entropy PE=H
        #loss = np.mean(entropy_vals) # should we calculate the average here ??
        return  loss

    def mutual_information_loss(self,model, X, y, num_samples=10, reduction='mean'):
        """Calculates the entropy loss of a probability distribution."""
        probs = None
        mean_entropy = None
        #entropy H
        for _ in range(num_samples):
            #MC dropout
            if hasattr(model, "enable_dropout"):
                model.enable_dropout()
            softmax_output = F.softmax(model(X), dim=1)

            if probs is None: probs = torch.zeros_like(softmax_output)
            probs = probs + softmax_output

            _entropy = entropy(softmax_output)
            if mean_entropy is None: mean_entropy = torch.zeros_like(_entropy)
            mean_entropy = mean_entropy + _entropy

            #probs = (probs+F.softmax(output, dim=1)) if probs is not None else F.softmax(output, dim=1) #probs = torch.zeros(output.shape).cuda()
            #mean_entropy = (mean_entropy+entropy(output)) if mean_entropy is not None else entropy(output) #: mean_entropy = torch.zeros(_entropy.shape).cuda()
            
            del softmax_output
            torch.cuda.empty_cache()

        probs /= float(num_samples)
        mask = probs == 0  # Create a mask of zero values
        probs[mask] = 10e-20  # Replace zero values with small_value

        log_probs = torch.log(probs)
        entropy_vals = -torch.sum(probs * log_probs, dim=1)# predictive entropy PE=H

        mean_entropy /= float(num_samples)
        mutual_information_vals = entropy_vals - mean_entropy

        if Normalize_entropy:
            num_clases = torch.Tensor([probs.size(dim=1)]).to(mutual_information_vals.device) 
            entropy_max = torch.log(num_clases)
            mutual_information_vals = torch.div(mutual_information_vals, entropy_max) # normalized mutual information 
        #mutual_information is maximum when the second term is 0 and the first is maxinum entropy (uniform distirbution)

        if reduction=='mean':
            loss = torch.mean(mutual_information_vals)  #mean mutial information
        elif reduction=='sum':
            loss = torch.sum(mutual_information_vals)  #total mutial information
        else:
            loss = mutual_information_vals


        
        return  loss


class CrossEntropy_Uncertainty_Loss(Uncertainty):
    # LOSS_MIN_CROSSENT_UNC  = 1 # minimizes cross_entropy_loss and uncertaint
    def __init__(self):
        super(CrossEntropy_Uncertainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):

        if standardization: 
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.std(loss_ce)
        
            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc = loss_unc/torch.std(loss_unc)

            loss_val = loss_ce + loss_unc
            loss_val = loss_val/torch.std(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc)

        elif normalization:
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce_max, loss_ce_min = loss_ce.max(), loss_ce.min() 
            #loss_ce = (loss_ce-loss_ce_min)/(loss_ce_max-loss_ce_min)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc_max, loss_unc_min = loss_unc.max(), loss_unc.min() 
            #loss_unc = (loss_unc-loss_unc_min)/(loss_unc_max-loss_unc_min)

            loss_val = loss_ce + loss_unc
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return torch.mean(loss_val) 
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 
        
        elif normalization_sum:
        
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.sum(loss_ce)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            #loss_unc = loss_unc/torch.sum(loss_unc)
            
            loss_val = loss_ce + loss_unc
            loss_val = loss_val/torch.sum(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = nn.CrossEntropyLoss()(y_pred,y)+self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = nn.CrossEntropyLoss()(y_pred,y)+self.mutual_information_loss(model, X, y, num_samples)

        return loss_val


class CrossEntropy_Certainty_Loss(Uncertainty):
    #LOSS_MIN_CROSSENT_MAX_UNC = 2 # minimize cross_entropy_loss - uncertainty = 
    # minimize cross_entropy_loss and  maximize uncertainty
    #  minimize cross_entropy_loss and  certainty  
    def __init__(self):
        super(CrossEntropy_Certainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):

        if standardization: 
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.std(loss_ce)
        
            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc = loss_unc/torch.std(loss_unc)

            loss_val = loss_ce - loss_unc
            loss_val = loss_val/torch.std(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc)

        elif normalization:
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce_max, loss_ce_min = loss_ce.max(), loss_ce.min() 
            #loss_ce = (loss_ce-loss_ce_min)/(loss_ce_max-loss_ce_min)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc_max, loss_unc_min = loss_unc.max(), loss_unc.min() 
            #loss_unc = (loss_unc-loss_unc_min)/(loss_unc_max-loss_unc_min)

            loss_val = loss_ce - loss_unc
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return torch.mean(loss_val) 
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 
        
        elif normalization_sum:
        
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.sum(loss_ce)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            #loss_unc = loss_unc/torch.sum(loss_unc)
            
            loss_val = loss_ce - loss_unc
            loss_val = loss_val/torch.sum(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = nn.CrossEntropyLoss()(y_pred,y)-self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = nn.CrossEntropyLoss()(y_pred,y)-self.mutual_information_loss(model, X, y, num_samples)

        return loss_val


class Certainty_Loss(Uncertainty):
    #LOSS_MAX_UNC = 4 # minimize -uncertainty = 
    #  = maximize uncertainty = minimize certainty  
    def __init__(self):
        super(Certainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):
        
        if standardization: 
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            loss_val = loss_val/torch.std(loss_val)
            return -torch.mean(loss_val)

        elif normalization:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return -torch.mean(loss_val) 
        
        elif normalization_sum:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            
            #print(loss_val)
            loss_val = loss_val/torch.sum(loss_val)
            return -torch.mean(loss_val) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples)

        return -loss_val


class Uncertainty_Loss(Uncertainty):
    #LOSS_MIN_UNC = 3 # minimize -uncertainty = maximize uncertainty 
    def __init__(self):
        super(Uncertainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):
        if standardization: 
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            loss_val = loss_val/torch.std(loss_val)
            return torch.mean(loss_val)

        elif normalization:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return torch.mean(loss_val) 
        
        elif normalization_sum:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            
            #print(loss_val)
            loss_val = loss_val/torch.sum(loss_val)
            return torch.mean(loss_val) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples)

        return loss_val


class CrossEntropy_Loss(nn.Module):
    #LOSS_MIN_CROSSENT= 0 # minimize cross_entropy_loss - uncertainty = 
    # minimize cross_entropy_loss 
    def __init__(self):
        super(CrossEntropy_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=0):
        if standardization:  
            _loss = nn.CrossEntropyLoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val = loss_val/torch.std(loss_val)
            return torch.mean(loss_val)

        elif normalization:
            _loss = nn.CrossEntropyLoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)
            return torch.mean(loss_val)    

        elif normalization_sum:
            _loss = nn.CrossEntropyLoss(reduction='none')
            loss_val = _loss(y_pred, y)
            #print(loss_val)

            loss_val = loss_val/torch.sum(loss_val)
            return torch.mean(loss_val)          

        #if dynamic_weights:
        #    _loss = nn.CrossEntropyLoss(reduction='none')
        #   return _loss(y_pred, y)
             
        return nn.CrossEntropyLoss()(y_pred, y)
    

class BinaryCrossEntropy_Loss(nn.Module):
    #LOSS_MIN_CROSSENT= 0 # minimize cross_entropy_loss - uncertainty = 
    # minimize cross_entropy_loss 
    def __init__(self):
        super(BinaryCrossEntropy_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=0):
        if standardization:  
            _loss = nn.BCELoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val = loss_val/torch.std(loss_val)
            return torch.mean(loss_val)

        elif normalization:
            _loss = nn.BCELoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)
            return torch.mean(loss_val)    

        elif normalization_sum:
            _loss = nn.BCELoss(reduction='none')
            loss_val = _loss(y_pred, y)
            #print(loss_val)

            loss_val = loss_val/torch.sum(loss_val)
            return torch.mean(loss_val)          

        #if dynamic_weights:
        #    _loss = nn.CrossEntropyLoss(reduction='none')
        #   return _loss(y_pred, y)
             
        return nn.BCELoss()(y_pred, y)


class ImageNetLoader(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        #print(self.data)
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            #print("tensor")

        x = self.data['x'][idx] 
        y = self.data['y'][idx] 

        if self.transform:
            x = self.transform(x)

        return x, y
        
    def __len__(self):
        #return self.data['x'].shape[0]
        return len(self.data['x'])


class SmallImagenet(VisionDataset):
    # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    #train_list = ['train_data_batch_{}'.format(i + 1) for i in range(1)]
    #print("change imageNet dataset batches")
    val_list = ['val_data']

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None, shuffle=False):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

        if shuffle:
            list_val = np.arange(len(self.data))
            random.shuffle(list_val)
            shuffled_data = np.array([self.data[key] for key in list_val])
            shuffled_target = np.array([self.targets[key] for key in list_val])
            self.data = shuffled_data
            self.targets = shuffled_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class BinaryCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, positive_class='cat', negative_class=None):
        # Class labels for CIFAR-10
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Choose two classes for binary classification
        #positive_class = positive_class
     
        self.cifar10 = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        self.positive_class = class_labels.index(positive_class)
        self.negative_class = class_labels.index(negative_class)
        self.data = []
        self.targets = []
        
        counter_positive_class = 0
        counter_negative_class = 0
        if negative_class is None:
            for data, target in self.cifar10:
                self.data.append(data)
                if target == self.positive_class:
                    counter_positive_class +=1
                    self.targets.append(1)  # 1 for positive class
                else:
                    counter_negative_class +=1
                    self.targets.append(0)  # 0 for negative class
        else:
            for data, target in self.cifar10:
                if target == self.positive_class:
                    counter_positive_class +=1
                    self.data.append(data)
                    self.targets.append(1)  # 1 for positive class
                elif target == self.negative_class:
                    self.data.append(data)
                    counter_negative_class +=1
                    self.targets.append(0)  # 0 for negative class

        print(positive_class + " class has size of " + str(counter_positive_class) + " and negtive class has size of " + str(counter_negative_class))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class CIFAR10C(VisionDataset):
    def __init__(self, root :str, name :str, transform=None, target_transform=None):
        
        corruptions = ['natural','gaussian_noise','shot_noise','speckle_noise','impulse_noise','defocus_blur','gaussian_blur','motion_blur','zoom_blur',\
                   'snow','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression','spatter','saturate','frost']

        assert name in corruptions

        dir  = root + '/CIFAR-10-C' 

        super(CIFAR10C, self).__init__(
            dir, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(dir, name + '.npy')
        target_path = os.path.join(dir, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


class CIFAR100C(VisionDataset):
    """CIFAR-100-C corruption dataset (same layout as CIFAR-10-C: one .npy per corruption)."""
    corruptions = ['natural', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                   'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                   'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

    def __init__(self, root: str, name: str, transform=None, target_transform=None):
        assert name in self.corruptions
        dir_path = os.path.join(root, 'CIFAR-100-C')
        super(CIFAR100C, self).__init__(dir_path, transform=transform, target_transform=target_transform)
        data_path = os.path.join(dir_path, name + '.npy')
        target_path = os.path.join(dir_path, 'labels.npy')
        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class DEUP():
    def __init__(self, trainloader, f_model, f_optim, device):
        self.trainloader = trainloader
        self.device=device

        self.f_predictor = f_model
        self.f_optimizer = f_optim
        self.e_predictor = ResNet(BasicBlock, [2, 2, 2, 2], 1, 18, 0.0).to(self.device) #create_network2(1, 1, 1024, 'relu', False, 5).to(self.device)
        self.e_optimizer =  optim.SGD(self.e_predictor.parameters(), lr=0.001, momentum=0.9)

        self.loss_fn=nn.CrossEntropyLoss(reduction='none')
        self.e_loss_fn=nn.MSELoss()

        self.percentage_valSet = 0.1


    def train(self, algorithm=None, epsilon=0.1, num_iter=20, alpha=0.01):
        
        #calidation set for calibration
        size = int(len(self.trainloader.data_val))
        #ids = random.choices(list(range(size)), k=int(size*0.1))
        ids = random.sample(range(size), int(size*self.percentage_valSet))
        subset_test = Subset(self.trainloader.data_val, ids)

        size_train = int(len(self.trainloader.data_train))
        ids_train = random.sample(range(size_train), int(size*self.percentage_valSet))
        subset_train = Subset(self.trainloader.data_train, ids_train)

        subset = ConcatDataset([subset_test, subset_train])
        #sub_loader = DataLoader(subset, batch_size=32 , shuffle=True) 
        
        #self.model_deup.fit_uncertainty_estimator_dataloader(self.trainloader.data_test, epochs=200, batch_size=self.trainloader.batch_size)
        if algorithm=='fgsm' or algorithm=='FGSM':
            self.fit_uncertainty_estimator_dataloader_adversarial(subset, epochs=50, batch_size=self.trainloader.batch_size_adv, algorithm='fgsm', epsilon=epsilon)
        elif algorithm=='pgd' or algorithm=='PGD':
            self.fit_uncertainty_estimator_dataloader_adversarial(subset, epochs=50, batch_size=self.trainloader.batch_size_adv, algorithm='fgsm', epsilon=epsilon, \
                                                                  num_iter=num_iter, alpha=alpha)
        else:
            self.fit_uncertainty_estimator_dataloader(subset, epochs=100, batch_size=self.trainloader.batch_size)
        #self.fit_uncertainty_estimator_dataloader(self.trainloader.data_train, epochs=100, batch_size=self.trainloader.batch_size)


    def fit_uncertainty_estimator_dataloader(self, data, epochs=None, batch_size=128, data_test=None):

        train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        if data_test is not None: test_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

        #building the dataset
        self.f_predictor.eval()
        loss_predictions_list = [] #torch.tensor([], device=self.device)
        features_list = [] #torch.tensor([], device=self.device)
        for features, target in train_loader: # target here is the real class
            features = features.to(self.device)
            target = target.to(self.device)

            target_pred = self.f_predictor(features)
            loss_base_model = self.loss_fn(target_pred, target) #.cpu()
            features_list += features.tolist()
            loss_predictions_list += loss_base_model.tolist()


        loader = DataLoader(TensorDataset(torch.tensor(features_list), torch.tensor(loss_predictions_list).unsqueeze(-1)), shuffle=True, batch_size=batch_size)

        #training the error predictor
        self.e_predictor.train()
        train_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            e_loss_sum, j = 0, 0
            for features, target in loader: # target here is the loss of the base model
                features, target = features.to(self.device), target.to(self.device)
                self.e_optimizer.zero_grad()

                predicted_uncertainties = self.e_predictor(features)
                e_loss = self.e_loss_fn(predicted_uncertainties, target)

                e_loss.backward()
                self.e_optimizer.step()

                epoch_losses.append(e_loss.item())
                e_loss_sum += e_loss.item()
                j+=1


            e_loss_t_sum, i = 0, 0
            if data_test is not None: 
                for features_t, target_t in test_loader:
                    features_t = features_t.to(self.device)
                    target_t = target_t.to(self.device)

                    target_pred_t = self.f_predictor(features_t)
                    loss_base_model_t = self.loss_fn(target_pred_t, target_t).unsqueeze(-1)
                    
                    pred = self.e_predictor(features_t)
                    e_loss_t = self.e_loss_fn(pred, loss_base_model_t)
                    e_loss_t_sum += e_loss_t.item()
                    i+=1
                
                print("train," + str(e_loss_sum/j) + ",test," + str(e_loss_t_sum/i))
            train_losses.append(np.mean(epoch_losses))

        return train_losses


    def fit_uncertainty_estimator_dataloader_adversarial(self, data, epochs=None, batch_size=128, data_test=None, algorithm='fgsm', epsilon=0.1, num_iter=20, alpha=0.01):

        train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        if data_test is not None: test_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

        #building the dataset
        self.f_predictor.eval()
        loss_predictions_list = [] #torch.tensor([], device=self.device)
        features_list = [] #torch.tensor([], device=self.device)
        for features, target in train_loader: # target here is the real class
            features = features.to(self.device)
            target = target.to(self.device)

            if algorithm == 'fgsm' or algorithm == 'FGSM':
                #Construct FGSM adversarial examples on the examples X
                delta = self.fgsm(features, target, epsilon)
            else:
                delta = self.pgd(features, target, epsilon, alpha, num_iter)

            X_input = features + delta

            target_pred = self.f_predictor(X_input)
            loss_base_model = self.loss_fn(target_pred, target) #.cpu()
            features_list += X_input.tolist()
            loss_predictions_list += loss_base_model.tolist()

        loader = DataLoader(TensorDataset(torch.tensor(features_list), torch.tensor(loss_predictions_list).unsqueeze(-1)), shuffle=True, batch_size=batch_size)

        #training the error predictor
        self.e_predictor.train()
        train_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            e_loss_sum, j = 0, 0
            for features, target in loader: # target here is the loss of the base model
                features, target = features.to(self.device), target.to(self.device)
                self.e_optimizer.zero_grad()

                predicted_uncertainties = self.e_predictor(features)
                e_loss = self.e_loss_fn(predicted_uncertainties, target)

                e_loss.backward()
                self.e_optimizer.step()

                epoch_losses.append(e_loss.item())
                e_loss_sum += e_loss.item()
                j+=1

            train_losses.append(np.mean(epoch_losses))

        return train_losses


    def predict(self, X):
        self.e_predictor.eval()
        return self.e_predictor(X) #model_deup._uncertainty(features=X) #.get_prediction_with_uncertainty(X)


    def fgsm(self, X, y, epsilon):
        """ Construct FGSM adversarial examples on the examples X"""
        delta = torch.zeros_like(X, requires_grad=True)
        X_input = X + delta
        y_pred = self.f_predictor(X_input)

        loss =  self.loss_fn(y_pred, y)
        loss.mean().backward()
            
        return epsilon * delta.grad.detach().sign()


    def pgd(self, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
        """ Construct PGD adversarial examples on the examples X"""
        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(X, requires_grad=True)
        delta.requires_grad = True
            
        for t in range(num_iter):
            X_input = X + delta
            y_pred = self.f_predictor(X_input)
            
            loss = self.loss_fn(y_pred, y)
            loss.mean().backward()

            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()

        return delta.detach()


class dataset():
    def __init__(self, dataset_name="mnist", batch_size = 100,  batch_size_adv = 100, imbalance="none", imb_factor=None, imb_seed=None, seed=None):
        batch_size_test = 32
        # the shuffle needs to be false for the DataLoader to more easily store the IDs of wrong classified inputs 
        self.batch_size = batch_size
        self.batch_size_adv = batch_size_adv
        self._dl_seed = seed  # for reproducible DataLoader shuffle + worker RNG (ImageNet etc.)
        
        if dataset_name == "mnist":
            self.num_classes = 10
            self.data_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
            self.data_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
            self.data_val = self.data_test


            # ---- DataLoader performance knobs (ImageNet) ----
            DL_WORKERS = int(os.environ.get("DL_WORKERS", "8"))  # try 4/8/16 on Colab
            DL_WORKERS = max(DL_WORKERS, 0)

            DL_KW = dict(
                pin_memory=True,
                num_workers=DL_WORKERS,
                persistent_workers=(DL_WORKERS > 0),
            )
            if DL_WORKERS > 0:
                DL_KW["prefetch_factor"] = 2

            self.train_loader = DataLoader(
                self.data_train,
                batch_size=batch_size,
                shuffle=True,
                **DL_KW,
            ) if batch_size > 0 else None

            self.trainAvd_loader = DataLoader(
                self.data_train,
                batch_size=batch_size_adv,
                shuffle=True,
                **DL_KW,
            ) if batch_size_adv > 0 else None

            self.test_loader = DataLoader(
                self.data_test,
                batch_size=batch_size_test,
                shuffle=False,
                **DL_KW,
            )
            #self.val_loader = self.test_loader


        elif dataset_name ==  "cifar10":
            self.num_classes = 10
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),])

            self.data_train = datasets.CIFAR10("../data", train=True, download=True, transform=train_transform)
            self.data_test = datasets.CIFAR10("../data", train=False, download=True, transform=test_transform)
            self.data_val = self.data_test
            if (imbalance or "").strip().lower() in ("exp", "longtail") and imb_factor is not None:
                lt_idx = _longtail_train_indices(self.data_train, 10, float(imb_factor), imb_seed)
                if lt_idx is not None:
                    self.data_train = Subset(self.data_train, lt_idx)
                    print(f"[DATA] CIFAR10 long-tail: imb_factor={imb_factor}, train samples={len(lt_idx)}", flush=True)

            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)
            #self.val_loader = self.test_loader


        elif dataset_name ==  "cifar10-c":
            self.num_classes = 10
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),])


            self.data_train = datasets.CIFAR10("../data", train=True, download=True, transform=train_transform)
            self.data_val = datasets.CIFAR10("../data", train=False, download=True, transform=test_transform)
            self.data_test = CIFAR10C("../data", name='gaussian_noise', transform=test_transform)
            
            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)
            #self.val_loader = DataLoader(self.data_val, batch_size = batch_size_test, shuffle=False)



        elif dataset_name ==  "binaryCifar10":
            self.num_classes = 2
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),])

            positive_class='cat'
            negative_class='dog'
            self.data_train = BinaryCIFAR10("../data", train=True, transform=train_transform, positive_class=positive_class, negative_class=negative_class)
            self.data_test = BinaryCIFAR10("../data", train=False, transform=test_transform, positive_class=positive_class, negative_class=negative_class)
            self.data_val = self.data_test

            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)
            #self.val_loader = self.test_loader


        elif dataset_name ==  "cifar100":
            self.num_classes = 100
            cifar100_mean = (0.5071, 0.4867, 0.4408)
            cifar100_std = (0.2675, 0.2565, 0.2761)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar100_mean, cifar100_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar100_mean, cifar100_std),])

            self.data_train = datasets.CIFAR100("../data", train=True, download=True, transform=train_transform)
            self.data_test = datasets.CIFAR100("../data", train=False, download=True, transform=test_transform)
            self.data_val = self.data_test
            if (imbalance or "").strip().lower() in ("exp", "longtail") and imb_factor is not None:
                lt_idx = _longtail_train_indices(self.data_train, 100, float(imb_factor), imb_seed)
                if lt_idx is not None:
                    self.data_train = Subset(self.data_train, lt_idx)
                    print(f"[DATA] CIFAR100 long-tail: imb_factor={imb_factor}, train samples={len(lt_idx)}", flush=True)

            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)
            #self.val_loader = self.test_loader


        elif dataset_name ==  "imageNet":
            self.num_classes = 1000
            if not imageNet_original:
                # SmallImageNet (32x32 or 64x64 pickle)
                print("[DATA] Using ImageNet 32/64 (SmallImageNet)", flush=True)
                root_dir = os.environ.get('IMAGENET_DS_ROOT', '../data/imageNet/')
                resolution = int(os.environ.get('IMAGENET_RES', '32'))
                classes = int(os.environ.get('IMAGENET_CLASSES', '1000'))
                print(f"[DATA] SmallImageNet root: {root_dir}  (resolution={resolution}, classes={classes})", flush=True)

                normalize = transforms.Normalize(mean=[0.4810,0.4574,0.4078], std=[0.2146,0.2104,0.2138])
                tf_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
                tf_test = transforms.Compose([transforms.ToTensor(),normalize,])

                self.data_train = SmallImagenet(root=root_dir, size=resolution, train=True, transform=tf_train, classes=range(classes), shuffle=True)
                self.data_test = SmallImagenet(root=root_dir, size=resolution, train=False, transform=tf_test, classes=range(classes))
                self.data_val = self.data_test
                _x, _ = self.data_train[0]
                _shape = _x.shape if hasattr(_x, "shape") else getattr(_x, "size", None)
                print(f"[DATA] ImageNet train sample shape: {_shape}  (expect (3,{resolution},{resolution}))", flush=True)
            else:
                # 224x224 ImageFolder (e.g. PSC shared ImageNet: dir with train/ and val/)
                print("[DATA] Using ImageNet 224x224 (ImageFolder)", flush=True)
                imagenet_root = os.environ.get("IMAGENET_ROOT", "").strip()
                if not imagenet_root:
                    data_dir_env = os.environ.get("DATA_DIR", "").strip() or os.environ.get("CEGS_DATA_DIR", "").strip()
                    if data_dir_env:
                        cand = os.path.join(data_dir_env, "imageNet")
                        if os.path.isdir(os.path.join(cand, "train")) and os.path.isdir(os.path.join(cand, "val")):
                            imagenet_root = cand
                if not imagenet_root:
                    imagenet_root = "../data/imageNet"
                traindir = os.path.join(imagenet_root, "train")
                valdir = os.path.join(imagenet_root, "val")
                if (not os.path.isdir(traindir)) or (not os.path.isdir(valdir)):
                    raise FileNotFoundError(
                        f"ImageNet train/val not found. Set IMAGENET_ROOT to the directory containing 'train' and 'val'. "
                        f"Got traindir={traindir}, valdir={valdir}"
                    )
                print(f"[DATA] ImageNet 224 root: {imagenet_root}", flush=True)
                crop_size = 224
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                _t_scan = time.time()
                print(f"[DATA] Building ImageFolder train from: {traindir}", flush=True)
                self.data_train = datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), normalize,
                ]))
                print(f"[DATA] Train ImageFolder ready: {len(self.data_train)} samples, took {time.time()-_t_scan:.1f}s", flush=True)
                _t_scan_val = time.time()
                print(f"[DATA] Building ImageFolder val from: {valdir}", flush=True)
                self.data_test = datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(), normalize,
                ]))
                print(f"[DATA] Val ImageFolder ready: {len(self.data_test)} samples, took {time.time()-_t_scan_val:.1f}s", flush=True)
                self.data_val = self.data_test

            # DataLoader: num_workers=0 = single-thread load = GPU starves, very slow. Use 4–10 for 32x32.
            DL_WORKERS = int(os.environ.get("DL_WORKERS", "10"))
            DL_WORKERS = max(DL_WORKERS, 0)
            print(f"[DATA] DataLoader: num_workers={DL_WORKERS} pin_memory=True persistent_workers={DL_WORKERS>0}", flush=True)
            DL_KW = dict(
                pin_memory=True,
                num_workers=DL_WORKERS,
                persistent_workers=(DL_WORKERS > 0),
            )
            if DL_WORKERS > 0:
                DL_KW["prefetch_factor"] = 2
            # Reproducibility: same seed => same batch order and same worker RNG (augmentations) across runs
            def _dl_kw_repro(offset=0):
                if self._dl_seed is None:
                    return DL_KW
                g = torch.Generator()
                g.manual_seed(self._dl_seed + offset)
                return {**DL_KW, "generator": g, "worker_init_fn": _seed_worker}

            self.train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True, **_dl_kw_repro(0)) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train, batch_size=batch_size_adv, shuffle=True, **_dl_kw_repro(1)) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size=batch_size_test, shuffle=False, **DL_KW)
            #self.val_loader = self.test_loader


        elif dataset_name ==  "svhn":
            self.num_classes = 10

            self.data_train = datasets.SVHN("../data", split='train', download=True, transform=transforms.ToTensor())
            self.data_test = datasets.SVHN("../data", split='test', download=True, transform=transforms.ToTensor())
            self.data_val = self.data_test
            if (imbalance or "").strip().lower() in ("exp", "longtail") and imb_factor is not None:
                lt_idx = _longtail_train_indices(self.data_train, 10, float(imb_factor), imb_seed)
                if lt_idx is not None:
                    self.data_train = Subset(self.data_train, lt_idx)
                    print(f"[DATA] SVHN long-tail: imb_factor={imb_factor}, train samples={len(lt_idx)}", flush=True)
            
            self.train_loader = DataLoader(self.data_train, batch_size = batch_size, shuffle=False, pin_memory=True,) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train, batch_size = batch_size_adv, shuffle=False, pin_memory=True,) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)
            #self.val_loader = self.test_loader


        else:
            raise("dataset not implementex")
        

    def load_data(self, idx, train=True, dir='../data'):
        if train:
            input_file = '/imageNet/train_data_batch_'
            #input_file = '/imageNet/Imagenet32_train/train_data_batch_'
            d = unpickle(dir+input_file+str(idx))
        else:
            input_file = '/imageNet/val_data'
            d = unpickle(dir+input_file)

        x = d['data']
        y = d['labels']

        y = [i-1 for i in y]

        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3))
        return x, y


    def update_trainLoader(self):
        DL_WORKERS = int(os.environ.get("DL_WORKERS", "8"))
        DL_WORKERS = max(DL_WORKERS, 0)
        DL_KW = dict(
            pin_memory=True,
            num_workers=DL_WORKERS,
            persistent_workers=(DL_WORKERS > 0),
        )
        if DL_WORKERS > 0:
            DL_KW["prefetch_factor"] = 2

        # keep shuffle=False here because code later relies on stable ordering/IDs
        self.train_loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=False,
            **DL_KW,
        ) if self.batch_size > 0 else None

        self.trainAvd_loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size_adv,
            shuffle=False,
            **DL_KW,
        ) if self.batch_size_adv > 0 else None
class trainModel():
    def __init__(self, device, half_prec=False, variants=None):
        self.device = device
        self.half_prec=half_prec

        #calibration FLAGS
        self.isCalibrated = False

        self.ToCalibrate = False 
        self.deup = False 
        self.deep_ensemble = False 
        self.cals = False 

        if variants == 'calibration': 
            self.ToCalibrate = True 
            print('Calibration enable')
        elif variants == 'deup':
            self.deup = True 
            print('deup enable')
        elif variants == 'ensemble':
            self.deep_ensemble = True 
            print('ensemble enable')
        elif variants == 'cals':
            self.cals = True 
            print('cals enable')
        
        self.LossInUse = LOSS_MIN_CROSSENT # LOSS_MIN_CROSSENT_UNC # 
        self.new_iterations = 30 #10 # 
        #####
        # this argument is used to specify the number of iteration of EUAT
        ####
        self.printTimes = False #True

        if self.half_prec:
            # we only used mixed precision in imageNet
            #self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O1")
            # Creates once at the beginning of training
            self.scaler = torch.cuda.amp.GradScaler()

        # ---- lightweight runtime diagnostics ----
        # Logs per-epoch loss-call latency (sampled) + epoch wall time to W&B.
        # Compare CE vs ECG/CEGS runs to show the gating/grad-scaling overhead is ~0 in practice.
        self.log_runtime = True
        self.rt_sample_every = 20  # measure every N loss calls; set 1 for full tracing
        self._rt_epoch_active = False
        self._rt_reset_epoch_buffers()
        # Print first batch shape once when DEBUG_IMAGENET_SHAPE=1
        self._debug_shape_printed = False
        return

    # -------------------------------
    # ECG scheduling utilities
    # -------------------------------
    def configure_ecg_schedule(
        self,
        schedule: str = "none",
        total_epochs: int = None,
        lam_start: float = None,
        lam_end: float = None,
        tau_start: float = None,
        tau_end: float = None,
        k_start: float = None,
        k_end: float = None,
        adapt_warmup: int = 10,
        # Scheme C (tau_target): control tau to keep active gate fraction near a target
        tau_target: float = 0.6,
        tau_lr: float = 0.10,
        tau_ema: float = 0.90,
        tau_deadzone: float = 0.02,
        tau_min: float = 0.0,
        tau_max: float = 0.99,
        adapt_window: int = 5,
    ):
        """Configure ECG parameter scheduling.

        schedule:
          - none: fixed (use --ecg_lam/--ecg_tau/--ecg_k)
          - linear: linearly interpolate start->end over [1..total_epochs]
          - cosine: cosine anneal start->end over [1..total_epochs]
          - adaptive: adjust params online based on recent error trend
          - tau_target: adjust tau online to keep gate active fraction near a target (Scheme C)

        When ecg_lam_rule in _AUTO_LAM_RULES: lam is only from gate_ema and delta (no schedule).
        ecg_lam_start/ecg_lam_end are not used for lam; ecg_lam_end in TSV is delta (or initial_delta for auto_d/auto_dw).
        auto_w: delta_eff = delta * min(1, epoch/5). auto_d/auto_dw: delta adapts from scale_p99 reference; auto_dw = 5-epoch warmup.
        """
        self.ecg_schedule = (schedule or "none").lower()
        self.ecg_total_epochs = int(total_epochs) if total_epochs is not None else None
        # --- Scheme C: tau_target controller params (used when ecg_schedule == 'tau_target') ---
        self.ecg_tau_target = float(tau_target)
        self.ecg_tau_lr = float(tau_lr)
        self.ecg_tau_ema = float(tau_ema)
        self.ecg_tau_deadzone = float(tau_deadzone)
        self.ecg_tau_min = float(tau_min)
        self.ecg_tau_max = float(tau_max)
        self._ecg_tau_target_ema_state = None

        # Base values come from current config
        self.ecg_lam_base = float(getattr(self, "ecg_lam", 1.0))
        self.ecg_tau_base = float(getattr(self, "ecg_tau", 0.7))
        self.ecg_k_base = float(getattr(self, "ecg_k", 10.0))

        # Start/end defaults (if not provided, keep constant). Auto-lambda: lam set per-batch from gate_ema
        if getattr(self, "ecg_lam_rule", None) in _AUTO_LAM_RULES:
            self.ecg_lam_start = None
            self.ecg_lam_end = None
        else:
            self.ecg_lam_start = self.ecg_lam_base if lam_start is None else float(lam_start)
            self.ecg_lam_end = self.ecg_lam_base if lam_end is None else float(lam_end)

        # tau: support "quantile" (fixed q), "auto_q" (scheduled q), "auto_q_ctrl" (P-controller q),
        #      and "auto_q_valley" (valley-detection q) via tau_rule
        if getattr(self, "ecg_tau_rule", None) in ("quantile", "auto_q", "auto_q_ctrl", "auto_q_valley"):
            self.ecg_tau_start = None
            self.ecg_tau_end = None
        else:
            self.ecg_tau_start = self.ecg_tau_base if tau_start is None else float(tau_start)
            self.ecg_tau_end = self.ecg_tau_base if tau_end is None else float(tau_end)

        self.ecg_k_start = self.ecg_k_base if k_start is None else float(k_start)
        self.ecg_k_end = self.ecg_k_base if k_end is None else float(k_end)

        self.ecg_adapt_warmup = int(adapt_warmup)
        self.ecg_adapt_window = max(1, int(adapt_window))
        self._ecg_metric_window = deque(maxlen=self.ecg_adapt_window)

        # Adaptive state (next-epoch params)
        self._ecg_adapt_state = {
            "lam": self.ecg_lam_base,
            "tau": self.ecg_tau_base,
            "k": self.ecg_k_base,
        }

        # Initialize visible params for scheduled modes
        if self.ecg_schedule in ("linear", "cosine"):
            if getattr(self, "ecg_lam_rule", None) not in _AUTO_LAM_RULES and self.ecg_lam_start is not None:
                self.ecg_lam = float(self.ecg_lam_start)
            if getattr(self, "ecg_tau_rule", None) not in ("quantile", "auto_q", "auto_q_ctrl", "auto_q_valley"):
                self.ecg_tau = float(self.ecg_tau_start) if self.ecg_tau_start is not None else self.ecg_tau_base
            self.ecg_k = float(self.ecg_k_start)
        elif self.ecg_schedule == "tau_target":
            # For tau_target, tau is controlled adaptively during training,
            # but lam/k may be scheduled. Initialize them to the start values
            # to avoid using base defaults in epoch 1.
            if hasattr(self, "ecg_lam_start"):
                self.ecg_lam = float(self.ecg_lam_start)
            if hasattr(self, "ecg_k_start"):
                self.ecg_k = float(self.ecg_k_start)
            if hasattr(self, "ecg_tau"):
                self.ecg_tau = float(self.ecg_tau)

        return

    def _ecg_schedule_progress(self, global_epoch: int) -> float:
        if self.ecg_total_epochs is None or self.ecg_total_epochs <= 1:
            return 1.0
        t = (float(global_epoch) - 1.0) / float(self.ecg_total_epochs - 1)
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return float(t)

    def _ecg_interp(self, start: float, end: float, t: float) -> float:
        return float(start + t * (end - start))

    def _ecg_cosine(self, start: float, end: float, t: float) -> float:
        # start at t=0, end at t=1
        return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))


    # -------------------------------
    # Runtime diagnostics utilities
    # -------------------------------
    def _rt_reset_epoch_buffers(self):
        self._rt_epoch_wall_start = None
        self._rt_epoch_wall_s = 0.0

        self._rt_epoch_imgs = 0
        self._rt_train_wall_start = None
        self._rt_train_wall_s = 0.0
        self._rt_eval_wall_start = None
        self._rt_eval_wall_s = 0.0

        self._rt_loss_call_idx = 0
        self._rt_loss_cpu_ms_sum = 0.0
        self._rt_loss_cpu_n = 0
        self._rt_loss_cpu_imgs = 0
        self._rt_loss_cuda_pairs = []
        self._rt_loss_cuda_ms_sum = 0.0
        self._rt_loss_cuda_n = 0
        self._rt_loss_cuda_imgs = 0

        self._rt_step_call_idx = 0
        self._rt_step_cpu_ms_sum = 0.0
        self._rt_step_cpu_n = 0
        self._rt_step_cpu_imgs = 0
        self._rt_step_cuda_pairs = []
        self._rt_step_cuda_ms_sum = 0.0
        self._rt_step_cuda_n = 0
        self._rt_step_cuda_imgs = 0

    def _mem_reset_peak_for_epoch(self) -> None:
        """Reset CUDA peak memory counters at epoch start (W&B diagnostics only)."""
        if not torch.cuda.is_available():
            return
        dev = getattr(self, "device", None)
        if dev is None or getattr(dev, "type", "") != "cuda":
            return
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass

    def _mem_cuda_mem_payload(self, synchronize: bool = True) -> Optional[dict]:
        """Current-epoch CUDA memory snapshot for W&B (MB); None if not applicable."""
        if not torch.cuda.is_available():
            return None
        dev = getattr(self, "device", None)
        if dev is None or getattr(dev, "type", "") != "cuda":
            return None
        if synchronize:
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        try:
            inv_mb = 1.0 / (1024.0 ** 2)
            return {
                "MEM/peak_allocated_mb": float(torch.cuda.max_memory_allocated(dev) * inv_mb),
                "MEM/peak_reserved_mb": float(torch.cuda.max_memory_reserved(dev) * inv_mb),
                "MEM/end_allocated_mb": float(torch.cuda.memory_allocated(dev) * inv_mb),
                "MEM/end_reserved_mb": float(torch.cuda.memory_reserved(dev) * inv_mb),
            }
        except Exception:
            return None

    def _rt_on_epoch_begin(self, global_epoch: int):
        # enable only when W&B is active and user didn't disable
        try:
            enabled = bool(getattr(self, "log_runtime", True)) and (wandb.run is not None)
        except Exception:
            enabled = False

        self._rt_enabled = enabled
        self._rt_epoch_active = bool(enabled)
        self._rt_epoch_wall_start = time.perf_counter() if enabled else None
        self._rt_reset_epoch_buffers()

        # keep epoch index (for logging)
        try:
            self._rt_epoch_idx = int(global_epoch)
        except Exception:
            self._rt_epoch_idx = None

    def _rt_on_epoch_end(self, global_epoch: int):
        if not getattr(self, "_rt_enabled", False):
            self._rt_epoch_active = False
            try:
                if wandb.run is not None:
                    mem = self._mem_cuda_mem_payload()
                    if mem:
                        wandb.log({"epoch": int(global_epoch), **mem}, step=int(global_epoch))
            except Exception:
                pass
            return

        # wall time
        try:
            if self._rt_epoch_wall_start is not None:
                self._rt_epoch_wall_s = float(time.perf_counter() - self._rt_epoch_wall_start)
        except Exception:
            self._rt_epoch_wall_s = 0.0

        # finalize CUDA loss timings (sync ONCE)
        if self._rt_loss_cuda_pairs:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            ms_sum = 0.0
            n = 0
            imgs = 0
            for ev0, ev1, bs in self._rt_loss_cuda_pairs:
                try:
                    ms = float(ev0.elapsed_time(ev1))  # milliseconds
                    ms_sum += ms
                    n += 1
                    imgs += int(bs)
                except Exception:
                    continue
            self._rt_loss_cuda_ms_sum = ms_sum
            self._rt_loss_cuda_n = n
            self._rt_loss_cuda_imgs = imgs

        # pick CPU/CUDA aggregates depending on device
        use_cuda = False
        try:
            use_cuda = (self.device is not None) and (getattr(self.device, "type", "") == "cuda")
        except Exception:
            use_cuda = False

        if use_cuda and self._rt_loss_cuda_n > 0:
            loss_ms_avg = self._rt_loss_cuda_ms_sum / float(self._rt_loss_cuda_n)
            loss_imgs = self._rt_loss_cuda_imgs
            loss_n = self._rt_loss_cuda_n
            backend = "cuda_event"
        else:
            loss_ms_avg = self._rt_loss_cpu_ms_sum / float(self._rt_loss_cpu_n) if self._rt_loss_cpu_n > 0 else float("nan")
            loss_imgs = self._rt_loss_cpu_imgs
            loss_n = self._rt_loss_cpu_n
            backend = "cpu_timer"

        # finalize step timings (sync ONCE)
        if self._rt_step_cuda_pairs:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            sms, sn, simgs = 0.0, 0, 0
            for ev0, ev1, bs in self._rt_step_cuda_pairs:
                try:
                    sms += float(ev0.elapsed_time(ev1))
                    sn += 1
                    simgs += int(bs)
                except Exception:
                    continue
            self._rt_step_cuda_ms_sum = sms
            self._rt_step_cuda_n = sn
            self._rt_step_cuda_imgs = simgs

        if use_cuda and self._rt_step_cuda_n > 0:
            step_ms_avg = self._rt_step_cuda_ms_sum / float(self._rt_step_cuda_n)
            step_imgs = self._rt_step_cuda_imgs
            step_n = self._rt_step_cuda_n
            step_backend = "cuda_event"
        else:
            step_ms_avg = self._rt_step_cpu_ms_sum / float(self._rt_step_cpu_n) if self._rt_step_cpu_n > 0 else float("nan")
            step_imgs = self._rt_step_cpu_imgs
            step_n = self._rt_step_cpu_n
            step_backend = "cpu_timer"

        # derived
        epoch_wall_s = float(self._rt_epoch_wall_s) if self._rt_epoch_wall_s else float("nan")
        epoch_imgs = int(getattr(self, "_rt_epoch_imgs", 0))
        train_wall_s = float(getattr(self, "_rt_train_wall_s", 0.0))
        eval_wall_s = float(getattr(self, "_rt_eval_wall_s", 0.0))
        epoch_img_per_s = (float(epoch_imgs) / train_wall_s) if (train_wall_s > 0 and epoch_imgs > 0) else float("nan")

        # log
        try:
            if wandb.run is not None:
                rt_minimal = bool(getattr(self, "rt_minimal_mode", False))
                log_d = {
                    "epoch": int(global_epoch),
                    "TIME/epoch_wall_s": epoch_wall_s,
                    "TIME/train_wall_s": train_wall_s,
                    "TIME/eval_wall_s": eval_wall_s,
                    "TIME/epoch_imgs": epoch_imgs,
                    "TIME/epoch_img_per_s": epoch_img_per_s,
                    "TIME/train_step_ms": float(step_ms_avg),
                    "TIME/train_step_n": int(step_n),
                    "TIME/train_step_imgs": int(step_imgs),
                    "TIME/train_step_backend": str(step_backend),
                    "TIME/train_step_sample_every": int(getattr(self, "rt_step_sample_every", 10)),
                }
                if not rt_minimal:
                    log_d.update({
                        "TIME/loss_call_ms": float(loss_ms_avg),
                        "TIME/loss_call_n": int(loss_n),
                        "TIME/loss_call_imgs": int(loss_imgs),
                        "TIME/backend": str(backend),
                        "TIME/rt_sample_every": int(getattr(self, "rt_sample_every", 0)),
                    })
                mem = self._mem_cuda_mem_payload()
                if mem:
                    log_d.update(mem)
                wandb.log(log_d, step=int(global_epoch))
        except Exception:
            pass

        self._rt_epoch_active = False

    def _rt_loss_timer_start(self, batch_size: int):
        # Only time during an "epoch scope" (between _rt_on_epoch_begin/end)
        if not getattr(self, "_rt_enabled", False) or not getattr(self, "_rt_epoch_active", False):
            return None

        self._rt_loss_call_idx += 1
        every = int(getattr(self, "rt_sample_every", 1) or 1)
        do_sample = (every <= 1) or (self._rt_loss_call_idx % every == 0)
        if not do_sample:
            return None

        # CUDA events (async) vs CPU timer
        use_cuda = False
        try:
            use_cuda = (self.device is not None) and (getattr(self.device, "type", "") == "cuda") and torch.cuda.is_available()
        except Exception:
            use_cuda = False

        if use_cuda:
            try:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                return ("cuda", ev0, int(batch_size))
            except Exception:
                pass

        # fallback
        return ("cpu", float(time.perf_counter()), int(batch_size))

    def _rt_loss_timer_end(self, token):
        if token is None:
            return

        kind = token[0]
        if kind == "cuda":
            ev0, bs = token[1], token[2]
            try:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self._rt_loss_cuda_pairs.append((ev0, ev1, int(bs)))
            except Exception:
                return
        else:
            t0, bs = float(token[1]), int(token[2])
            t1 = float(time.perf_counter())
            self._rt_loss_cpu_ms_sum += (t1 - t0) * 1000.0
            self._rt_loss_cpu_n += 1
            self._rt_loss_cpu_imgs += int(bs)

    def _rt_add_epoch_imgs(self, n):
        self._rt_epoch_imgs = getattr(self, "_rt_epoch_imgs", 0) + int(n)

    def _rt_train_begin(self):
        if getattr(self, "_rt_enabled", False):
            self._rt_train_wall_start = time.perf_counter()

    def _rt_train_end(self):
        if getattr(self, "_rt_enabled", False) and getattr(self, "_rt_train_wall_start", None) is not None:
            self._rt_train_wall_s = getattr(self, "_rt_train_wall_s", 0.0) + (time.perf_counter() - self._rt_train_wall_start)
            self._rt_train_wall_start = None

    def _rt_eval_begin(self):
        if getattr(self, "_rt_enabled", False):
            self._rt_eval_wall_start = time.perf_counter()

    def _rt_eval_end(self):
        if getattr(self, "_rt_enabled", False) and getattr(self, "_rt_eval_wall_start", None) is not None:
            self._rt_eval_wall_s = getattr(self, "_rt_eval_wall_s", 0.0) + (time.perf_counter() - self._rt_eval_wall_start)
            self._rt_eval_wall_start = None

    def _rt_step_timer_start(self, batch_size):
        if not getattr(self, "_rt_enabled", False) or not getattr(self, "_rt_epoch_active", False):
            return None
        self._rt_step_call_idx += 1
        every = int(getattr(self, "rt_step_sample_every", 10) or 10)
        if every > 1 and (self._rt_step_call_idx % every != 0):
            return None
        use_cuda = False
        try:
            use_cuda = (self.device is not None) and (getattr(self.device, "type", "") == "cuda") and torch.cuda.is_available()
        except Exception:
            pass
        if use_cuda:
            try:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                return ("cuda", ev0, int(batch_size))
            except Exception:
                pass
        return ("cpu", float(time.perf_counter()), int(batch_size))

    def _rt_step_timer_end(self, token):
        if token is None:
            return
        kind = token[0]
        if kind == "cuda":
            ev0, bs = token[1], token[2]
            try:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self._rt_step_cuda_pairs.append((ev0, ev1, int(bs)))
            except Exception:
                return
        else:
            t0, bs = float(token[1]), int(token[2])
            self._rt_step_cpu_ms_sum += (time.perf_counter() - t0) * 1000.0
            self._rt_step_cpu_n += 1
            self._rt_step_cpu_imgs += int(bs)

    def _ecg_on_epoch_begin(self, global_epoch: int):
        """Apply ECG scheduling before the epoch starts."""

        # --- CUDA memory (epoch window): reset peak stats before train/eval ---
        try:
            self._mem_reset_peak_for_epoch()
        except Exception:
            pass

        # --- ECG stats (epoch-level): reset accumulator ---
        self._ecg_stat_sum = {}
        self._ecg_stat_n = 0

        # --- Runtime stats (epoch-level) ---
        try:
            self._rt_on_epoch_begin(global_epoch)
        except Exception:
            pass

        # --- auto_q: scheduled quantile q (linear q_start -> q_end over ECG-active epochs only); log ECG/tau_q_cur ---
        try:
            if getattr(self, "ecg_tau_rule", None) == "auto_q":
                s1 = int(getattr(self, "stage1_epochs", 0))
                s2 = int(getattr(self, "stage2_epochs", 0))
                if s2 > 0:
                    # ECG-active = stage2 only: t=0 at first ECG epoch (global_epoch=s1+1), t=1 at last
                    t = (float(global_epoch) - (s1 + 1.0)) / max(1.0, float(s2) - 1.0)
                else:
                    # full ECG from epoch 1 (stage2_epochs=0): progress over full training
                    total = int(getattr(self, "ecg_total_epochs", s1)) or s1
                    t = (float(global_epoch) - 1.0) / max(1.0, float(total) - 1.0) if total > 1 else 1.0
                t = max(0.0, min(1.0, t))
                q_start = getattr(self, "ecg_tau_q_start", 0.6)
                q_end = getattr(self, "ecg_tau_q_end", 0.9)
                q_cur = q_start + (q_end - q_start) * t
                self.ecg_tau_quantile_cur = float(q_cur)
                if wandb.run is not None:
                    wandb.log({"epoch": int(global_epoch), "ECG/tau_q_cur": float(q_cur)}, step=int(global_epoch))
        except Exception:
            pass

        sched = getattr(self, "ecg_schedule", "none")
        if sched in (None, "none", "fixed"):
            return

        if sched in ("linear", "cosine"):
            t = self._ecg_schedule_progress(global_epoch)
            lam_auto = getattr(self, "ecg_lam_rule", None) in _AUTO_LAM_RULES
            tau_quantile = getattr(self, "ecg_tau_rule", None) in ("quantile", "auto_q", "auto_q_ctrl", "auto_q_valley")
            if lam_auto and tau_quantile:
                # both auto: only schedule k
                if sched == "linear":
                    self.ecg_k = self._ecg_interp(self.ecg_k_start, self.ecg_k_end, t)
                else:
                    self.ecg_k = self._ecg_cosine(self.ecg_k_start, self.ecg_k_end, t)
            elif lam_auto:
                # lam auto; schedule tau and k
                if sched == "linear":
                    if not tau_quantile:
                        self.ecg_tau = self._ecg_interp(self.ecg_tau_start, self.ecg_tau_end, t)
                    self.ecg_k = self._ecg_interp(self.ecg_k_start, self.ecg_k_end, t)
                else:
                    if not tau_quantile:
                        self.ecg_tau = self._ecg_cosine(self.ecg_tau_start, self.ecg_tau_end, t)
                    self.ecg_k = self._ecg_cosine(self.ecg_k_start, self.ecg_k_end, t)
            elif tau_quantile:
                # tau quantile; schedule lam and k
                if sched == "linear":
                    self.ecg_lam = self._ecg_interp(self.ecg_lam_start, self.ecg_lam_end, t)
                    self.ecg_k = self._ecg_interp(self.ecg_k_start, self.ecg_k_end, t)
                else:
                    self.ecg_lam = self._ecg_cosine(self.ecg_lam_start, self.ecg_lam_end, t)
                    self.ecg_k = self._ecg_cosine(self.ecg_k_start, self.ecg_k_end, t)
            else:
                if sched == "linear":
                    self.ecg_lam = self._ecg_interp(self.ecg_lam_start, self.ecg_lam_end, t)
                    self.ecg_tau = self._ecg_interp(self.ecg_tau_start, self.ecg_tau_end, t)
                    self.ecg_k = self._ecg_interp(self.ecg_k_start, self.ecg_k_end, t)
                else:
                    self.ecg_lam = self._ecg_cosine(self.ecg_lam_start, self.ecg_lam_end, t)
                    self.ecg_tau = self._ecg_cosine(self.ecg_tau_start, self.ecg_tau_end, t)
                    self.ecg_k = self._ecg_cosine(self.ecg_k_start, self.ecg_k_end, t)

        elif sched == "adaptive":
            st = getattr(self, "_ecg_adapt_state", None)
            if st is not None:
                if getattr(self, "ecg_lam_rule", None) not in _AUTO_LAM_RULES:
                    self.ecg_lam = float(st["lam"])
                self.ecg_tau = float(st["tau"])
                self.ecg_k = float(st["k"])

        elif sched == "tau_target":
            # In tau_target mode, tau is updated at epoch end by the controller.
            # But we still want lam/k to follow a smooth schedule (if start/end are provided)
            # and to be correctly applied from epoch 1.
            t = self._ecg_schedule_progress(global_epoch)
            # lam schedule (skip when auto-lambda)
            if getattr(self, "ecg_lam_rule", None) not in _AUTO_LAM_RULES and getattr(self, "ecg_lam_start", None) is not None and getattr(self, "ecg_lam_end", None) is not None:
                self.ecg_lam = self._ecg_interp(self.ecg_lam_start, self.ecg_lam_end, t)
            # k schedule
            if hasattr(self, "ecg_k_start") and hasattr(self, "ecg_k_end"):
                self.ecg_k = self._ecg_interp(self.ecg_k_start, self.ecg_k_end, t)
            # tau stays as the current controller value (self.ecg_tau)

        # Log current schedule values once per epoch (if wandb active)
        try:
            if wandb.run is not None:
                to_log = {
                    "epoch": int(global_epoch),
                    "ECG/lam": float(getattr(self, "ecg_lam", 0.0)),
                    "ECG/tau": float(getattr(self, "ecg_tau", 0.0)),
                    "ECG/k": float(getattr(self, "ecg_k", 0.0)),
                    "ECG/schedule_progress": float(self._ecg_schedule_progress(global_epoch)),
                    "ECG/schedule": str(sched),
                }
                if getattr(self, "ecg_conf_type", "pmax") == "pmax_temp":
                    to_log["ECG/ecg_gate_temp"] = float(getattr(self, "ecg_gate_temp", 1.5))
                if getattr(self, "ecg_lam_rule", None) in _AUTO_LAM_RULES:
                    to_log["ECG/gate_ema"] = float(getattr(self, "_ecg_gate_ema", 0.0))
                    to_log["ECG/lam_auto"] = float(getattr(self, "ecg_lam", 0.0))
                    if getattr(self, "ecg_lam_rule", None) == "auto_w":
                        delta = getattr(self, "ecg_lam_delta", 0.05)
                        to_log["ECG/delta_eff"] = float(delta * min(1.0, global_epoch / 5.0))
                    if getattr(self, "ecg_lam_rule", None) in ("auto_tr", "auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                        to_log["ECG/gate_mean_ema"] = float(getattr(self, "_ecg_gate_ema", 0.0))
                        to_log["ECG/gate_p99_ema"] = float(getattr(self, "_ecg_gate_p99_ema", 0.0))
                        to_log["ECG/conf_gate_active_frac_ema"] = float(getattr(self, "_ecg_active_frac_ema", 0.0))
                        to_log["ECG/lam_auto_raw_target"] = float(getattr(self, "_ecg_lam_auto_raw", 0.0))
                        to_log["ECG/lam_auto_smoothed"] = float(getattr(self, "_ecg_lam_auto_smoothed", 0.0))
                        to_log["ECG/lam_auto_after_guard"] = float(getattr(self, "_ecg_lam_auto_after_guard", 0.0))
                        to_log["ECG/tail_ratio_target"] = float(getattr(self, "ecg_tail_ratio_target", 3.0))
                        lam_cur = float(getattr(self, "ecg_lam", 0.0))
                        gme = float(getattr(self, "_ecg_gate_ema", 0.0))
                        g99 = float(getattr(self, "_ecg_gate_p99_ema", 0.0))
                        if gme > 1e-12:
                            to_log["ECG/tail_ratio_est"] = (1.0 + lam_cur * g99) / (1.0 + lam_cur * gme)
                        else:
                            to_log["ECG/tail_ratio_est"] = 0.0
                    if getattr(self, "ecg_lam_rule", None) in ("auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                        to_log["ECG/scale_p99_ema"] = float(getattr(self, "_ecg_scale_p99_ema_tr", 0.0))
                        to_log["ECG/lam_auto_base"] = float(getattr(self, "_ecg_lam_auto_smoothed", 0.0))
                        to_log["ECG/lam_sustain_target"] = float(getattr(self, "_ecg_lam_sustain_target", 0.0))
                        to_log["ECG/lam_sustain_ema"] = float(getattr(self, "_ecg_lam_sustain_ema", 0.0))
                        to_log["ECG/lam_auto_after_sustain"] = float(getattr(self, "_ecg_lam_auto_after_guard", 0.0))
                    if getattr(self, "ecg_lam_rule", None) in ("auto_tr_autocap", "auto_tr_autocap_gate"):
                        to_log["ECG/lam_cap_cur"] = float(getattr(self, "_ecg_lam_cap_cur", 0.0))
                        to_log["ECG/lam_cap_min"] = float(getattr(self, "_ecg_lam_cap_min", 0.0))
                        to_log["ECG/lam_cap_hit_ema"] = float(getattr(self, "_ecg_lam_cap_hit_ema", 0.0))
                        to_log["ECG/lam_inner_before_cap"] = float(getattr(self, "_ecg_lam_inner_before_cap", 0.0))
                        to_log["ECG/lam_auto_after_cap"] = float(getattr(self, "ecg_lam", 0.0))
                        to_log["ECG/cap_up_pressure"] = float(getattr(self, "_ecg_cap_up_pressure", 0.0))
                        to_log["ECG/cap_down_pressure"] = float(getattr(self, "_ecg_cap_down_pressure", 0.0))
                    if getattr(self, "ecg_lam_rule", None) == "auto_tr_autocap_gate":
                        to_log["ECG/gate_floor"] = 0.12
                        to_log["ECG/gate_narrow_deficit"] = float(getattr(self, "_ecg_gate_narrow_deficit", 0.0))
                        to_log["ECG/q_cur_base"] = float(getattr(self, "_ecg_gate_q_base", 0.0))
                        q_corr = float(getattr(self, "_ecg_gate_q_correction", 0.0))
                        to_log["ECG/q_cur_after_gate_correction"] = float(getattr(self, "_ecg_gate_q_base", 0.0)) - q_corr
                        to_log["ECG/tau_q_correction"] = q_corr
                    # auto_d/auto_dw: delta_cur, delta_eff, scale_p99_ema logged only at epoch end
                wandb.log(to_log, step=int(global_epoch))
        except Exception:
            pass

        return

    def _ecg_on_epoch_end(self, global_epoch: int, metric: float = None):
        """Update adaptive ECG schedule after the epoch ends.

        metric: use test error (lower is better).
        """

        # --- ECG stats (epoch-level): log averaged gate stats ---
        try:
            if wandb.run is not None and getattr(self, "_ecg_stat_n", 0) > 0:
                avg = {f"ECG/{k}": (v / float(self._ecg_stat_n)) for k, v in getattr(self, "_ecg_stat_sum", {}).items()}
                if avg:
                    wandb.log({"epoch": int(global_epoch), **avg}, step=int(global_epoch))
        except Exception:
            pass

        # --- Auto-delta (auto_d/auto_dw): update delta once per epoch from scale_p99 reference; log delta_cur, delta_eff, scale_p99_ema ---
        try:
            if getattr(self, "ecg_lam_rule", None) in ("auto_d", "auto_dw") and getattr(self, "_ecg_stat_n", 0) > 0:
                s = getattr(self, "_ecg_stat_sum", {}) or {}
                if "scale_p99_after_norm" in s:
                    import math
                    scale_p99_epoch_avg = float(s["scale_p99_after_norm"]) / float(self._ecg_stat_n)
                    beta_p99 = getattr(self, "ecg_auto_d_beta_p99", 0.9)
                    if getattr(self, "_ecg_scale_p99_ema", None) is None:
                        self._ecg_scale_p99_ema = scale_p99_epoch_avg
                    else:
                        self._ecg_scale_p99_ema = beta_p99 * self._ecg_scale_p99_ema + (1.0 - beta_p99) * scale_p99_epoch_avg
                    target = getattr(self, "ecg_auto_d_target_p99", 1.55)
                    eta = getattr(self, "ecg_auto_d_eta", 0.05)
                    delta_min = getattr(self, "ecg_auto_d_delta_min", 0.01)
                    delta_max = getattr(self, "ecg_auto_d_delta_max", 0.20)
                    delta_cur = getattr(self, "_ecg_delta_cur", getattr(self, "ecg_lam_delta", 0.05))
                    delta_cur = delta_cur * math.exp(eta * (target - self._ecg_scale_p99_ema))
                    delta_cur = max(delta_min, min(delta_max, delta_cur))
                    self._ecg_delta_cur = delta_cur
                    self._ecg_delta_eff = delta_cur
                    if wandb.run is not None:
                        wandb.log({
                            "epoch": int(global_epoch),
                            "ECG/delta_cur": float(delta_cur),
                            "ECG/delta_eff": float(self._ecg_delta_eff),
                            "ECG/scale_p99_ema": float(self._ecg_scale_p99_ema),
                        }, step=int(global_epoch))
        except Exception:
            pass

        # --- Runtime stats (epoch-level): log once per epoch ---
        try:
            self._rt_on_epoch_end(global_epoch)
        except Exception:
            pass

        sched = getattr(self, "ecg_schedule", "none")

        # --- Scheme C: tau_target controller (does not need metric) ---
        if sched == "tau_target":
            try:
                n = int(getattr(self, "_ecg_stat_n", 0))
                if n > 0:
                    s = getattr(self, "_ecg_stat_sum", {}) or {}
                    if "conf_gate_active_frac" in s:
                        active = float(s["conf_gate_active_frac"]) / float(n)

                        # optional EMA smoothing
                        beta = float(getattr(self, "ecg_tau_ema", 0.0))
                        if beta > 0.0:
                            prev = getattr(self, "_ecg_tau_target_ema_state", None)
                            if prev is None:
                                ema = active
                            else:
                                ema = beta * float(prev) + (1.0 - beta) * active
                            self._ecg_tau_target_ema_state = float(ema)
                            active_used = float(ema)
                        else:
                            active_used = float(active)

                        target = float(getattr(self, "ecg_tau_target", 0.6))
                        err = active_used - target

                        deadzone = float(getattr(self, "ecg_tau_deadzone", 0.0))
                        if abs(err) >= deadzone:
                            tau = float(getattr(self, "ecg_tau", getattr(self, "ecg_tau_base", 0.7)))
                            lr = float(getattr(self, "ecg_tau_lr", 0.1))
                            tau_new = tau + lr * err
                            tau_min = float(getattr(self, "ecg_tau_min", 0.0))
                            tau_max = float(getattr(self, "ecg_tau_max", 0.99))
                            tau_new = float(min(max(tau_new, tau_min), tau_max))
                            self.ecg_tau = tau_new

                        # log controller behavior
                        try:
                            if wandb.run is not None:
                                wandb.log(
                                    {
                                        "epoch": int(global_epoch),
                                        "ECG/tau_target_active_frac": float(active_used),
                                        "ECG/tau_target_raw_active_frac": float(active),
                                        "ECG/tau_target": float(target),
                                        "ECG/tau_target_err": float(err),
                                        "ECG/tau_after": float(getattr(self, "ecg_tau", 0.0)),
                                    },
                                    step=int(global_epoch),
                                )
                        except Exception:
                            pass
            except Exception:
                pass
            
            # Also allow lam/k to follow the configured start->end schedule while tau is controlled.
            # Auto-lam: lam is only from gate_ema + delta; do not overwrite with schedule.
            try:
                if getattr(self, "ecg_total_epochs", None) is not None:
                    t = self._ecg_schedule_progress(int(global_epoch))
                    if getattr(self, "ecg_lam_rule", None) not in _AUTO_LAM_RULES:
                        lam_new = (1.0 - t) * float(getattr(self, "ecg_lam_start", getattr(self, "ecg_lam", 0.0))) + t * float(getattr(self, "ecg_lam_end", getattr(self, "ecg_lam", 0.0)))
                        self.ecg_lam = float(lam_new)
                    k_new   = (1.0 - t) * float(getattr(self, "ecg_k_start", getattr(self, "ecg_k", 0.0)))   + t * float(getattr(self, "ecg_k_end", getattr(self, "ecg_k", 0.0)))
                    self.ecg_k = float(k_new)
                    # log
                    try:
                        if wandb.run is not None:
                            wandb.log({"epoch": int(global_epoch), "ECG/lam_after": float(self.ecg_lam), "ECG/k_after": float(self.ecg_k)}, step=int(global_epoch))
                    except Exception:
                        pass
            except Exception:
                pass
            return

        # --- auto_q_ctrl: P controller on quantile q to keep active gate fraction near target ---
        if getattr(self, "ecg_tau_rule", None) == "auto_q_ctrl":
            try:
                n = int(getattr(self, "_ecg_stat_n", 0))
                if n > 0:
                    s = getattr(self, "_ecg_stat_sum", {}) or {}
                    if "conf_gate_active_frac" in s:
                        active = float(s["conf_gate_active_frac"]) / float(n)

                        # EMA smoothing (reuses ecg_tau_ema)
                        beta = float(getattr(self, "ecg_tau_ema", 0.9))
                        prev = getattr(self, "_ecg_auto_q_ctrl_ema", None)
                        ema = active if prev is None else beta * float(prev) + (1.0 - beta) * active
                        self._ecg_auto_q_ctrl_ema = float(ema)

                        target = float(getattr(self, "ecg_tau_q_ctrl_target", 0.3))
                        err = ema - target  # positive: too many active gates → q too low → raise q

                        deadzone = float(getattr(self, "ecg_tau_deadzone", 0.02))
                        if abs(err) >= deadzone:
                            q_cur = float(getattr(self, "ecg_tau_quantile_cur",
                                                   getattr(self, "ecg_tau_q_start", 0.6)))
                            lr_q  = float(getattr(self, "ecg_tau_lr", 0.05))
                            q_new = q_cur + lr_q * err
                            q_min = float(getattr(self, "ecg_tau_min", 0.1))
                            q_max = float(getattr(self, "ecg_tau_max", 0.99))
                            self.ecg_tau_quantile_cur = float(min(max(q_new, q_min), q_max))

                        try:
                            if wandb.run is not None:
                                wandb.log({
                                    "epoch": int(global_epoch),
                                    "ECG/auto_q_ctrl_active_frac":     float(ema),
                                    "ECG/auto_q_ctrl_raw_active_frac": float(active),
                                    "ECG/auto_q_ctrl_target_frac":     float(target),
                                    "ECG/auto_q_ctrl_err":             float(err),
                                    "ECG/tau_q_cur":                   float(getattr(self, "ecg_tau_quantile_cur", 0.0)),
                                }, step=int(global_epoch))
                        except Exception:
                            pass
            except Exception:
                pass

        # --- auto_q_valley: find confidence distribution valley each epoch, use it as q ---
        if getattr(self, "ecg_tau_rule", None) == "auto_q_valley":
            try:
                warmup = int(getattr(self, "ecg_tau_valley_warmup", 5))
                if int(global_epoch) > warmup:
                    s = getattr(self, "_ecg_stat_sum", {}) or {}
                    hist = s.get("_conf_hist", None)
                    if hist is not None and len(hist) >= 3:
                        import math as _math
                        n_bins = len(hist)
                        # Gaussian smooth (kernel width = ecg_tau_valley_smooth bins, default 3)
                        smooth = max(1, int(getattr(self, "ecg_tau_valley_smooth", 3)))
                        smoothed = [0.0] * n_bins
                        total_counts = sum(hist) or 1.0
                        for i in range(n_bins):
                            w_sum = 0.0
                            c_sum = 0.0
                            for d in range(-smooth * 2, smooth * 2 + 1):
                                j = i + d
                                if 0 <= j < n_bins:
                                    w = _math.exp(-0.5 * (d / max(smooth, 1)) ** 2)
                                    c_sum += hist[j] * w
                                    w_sum += w
                            smoothed[i] = c_sum / (w_sum or 1.0)

                        # Find first local minimum between bins.
                        # Skip conf < 0.1 (bottom 10% of bins) to avoid edge noise.
                        # Require the candidate to be below 80% of the global max.
                        _bin_min = max(1, n_bins // 10)  # skip conf < 0.1
                        valley_bin = None
                        for i in range(_bin_min, n_bins - 1):
                            if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
                                if smoothed[i] < max(smoothed) * 0.8:
                                    valley_bin = i
                                    break

                        if valley_bin is not None:
                            # Convert bin index to conf value
                            valley_conf = (valley_bin + 0.5) / n_bins  # bin centre in [0,1]
                            # Convert conf value to quantile using the accumulated histogram
                            cum = 0.0
                            q_valley = 0.5  # fallback
                            for i, c in enumerate(hist):
                                cum += c
                                if cum / total_counts >= valley_conf:
                                    q_valley = (i + 0.5) / n_bins
                                    break
                            # EMA smooth (reuses ecg_tau_ema)
                            beta = float(getattr(self, "ecg_tau_ema", 0.9))
                            prev_q = float(getattr(self, "ecg_tau_quantile_cur",
                                                    getattr(self, "ecg_tau_q_start", 0.6)))
                            q_new = beta * prev_q + (1.0 - beta) * q_valley
                            q_min = float(getattr(self, "ecg_tau_min", 0.1))
                            q_max = float(getattr(self, "ecg_tau_max", 0.99))
                            self.ecg_tau_quantile_cur = float(min(max(q_new, q_min), q_max))

                            try:
                                if wandb.run is not None:
                                    wandb.log({
                                        "epoch": int(global_epoch),
                                        "ECG/auto_q_valley_conf":    float(valley_conf),
                                        "ECG/auto_q_valley_bin":     int(valley_bin),
                                        "ECG/auto_q_valley_q_raw":   float(q_valley),
                                        "ECG/tau_q_cur":             float(self.ecg_tau_quantile_cur),
                                    }, step=int(global_epoch))
                            except Exception:
                                pass
            except Exception:
                pass

        if sched != "adaptive":
            return

        if metric is None:
            return

        try:
            m = float(metric)
        except Exception:
            return

        self._ecg_metric_window.append(m)

        # Warmup: do not adapt
        if int(global_epoch) <= int(getattr(self, "ecg_adapt_warmup", 0)):
            return
        if len(self._ecg_metric_window) < int(getattr(self, "ecg_adapt_window", 1)):
            return

        # Trend over the window
        delta = float(self._ecg_metric_window[-1] - self._ecg_metric_window[0])

        # Small deadzone
        if abs(delta) < 1e-4:
            return

        # If error worsens (delta>0): strengthen ECG; else relax a bit.
        direction = 1.0 if delta > 0.0 else -1.0

        lam_step = 0.05 * max(1.0, abs(float(getattr(self, "ecg_lam_base", 1.0))))
        tau_step = 0.01
        k_step = 0.5

        st = self._ecg_adapt_state
        st["lam"] = float(min(max(st["lam"] + direction * lam_step, 0.0), 5.0))
        st["tau"] = float(min(max(st["tau"] - direction * tau_step, 0.0), 0.99))
        st["k"] = float(min(max(st["k"] + direction * k_step, 0.1), 100.0))

        try:
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": int(global_epoch),
                        "ECG/adapt_delta_err": float(delta),
                        "ECG/adapt_next_lam": float(st["lam"]),
                        "ECG/adapt_next_tau": float(st["tau"]),
                        "ECG/adapt_next_k": float(st["k"]),
                    },
                    step=int(global_epoch),
                )
        except Exception:
            pass

        return

    @torch.no_grad()
    def compute_train_ce_err(self, train_loader, model):
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_err = 0.0
        total_n = 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = model(X)

            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            total_err  += (logits.argmax(dim=1) != y).sum().item()
            total_n    += y.size(0)

        if was_training:
            model.train()

        return total_loss / total_n, total_err / total_n

    def writeResult(self, filename, data):
        trainTime, train_err, train_loss = data[0]
        testTime, test_err, test_loss = data[1]
        advTestTime_pgd, adv_err_pgd, adv_loss_pgd = data[2]

        str1 = "total_time=" + str(trainTime) + ";"
        str1 += "\ntraining_error=" + str(train_err) + ";"
        str1 += "\ntraining_loss=" + str(train_loss) + ";"

        str1 += "\ntesting_time=" + str(testTime) + ";"
        str1 += "\ntest_error=" + str(test_err) + ";"
        str1 += "\ntest_loss=" + str(test_loss) + ";"

        str1 += "\navd_test_time_pgd=" + str(advTestTime_pgd) + ";"
        str1 += "\nadversarial_error_pgd=" + str(adv_err_pgd) + ";"
        str1 += "\nadversarial_loss_pgd=" + str(adv_loss_pgd) + ";"

        if len(data) == 4:
            advTestTime_fgsm, adv_err_fgsm, adv_loss_fgsm = data[3]
            str1 += "\navd_test_time_fgsm=" + str(advTestTime_fgsm) + ";"
            str1 += "\nadversarial_error_fgsm=" + str(adv_err_fgsm) + ";"
            str1 += "\nadversarial_loss_fgsm=" + str(adv_loss_fgsm) + ";"

        f = open(filename, "w")
        f.write(str1)
        f.close()

    def saveModel(self, isBest, state, filename, epoch):
        save_dir = os.path.join(os.path.dirname(__file__), "models")  # /content/ECG_Loss/models
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}_epoch{epoch}.pt")
        torch.save(state, save_path)

    def LoadModel(self, model, opt, filename):
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        ckpt_path = os.path.join(model_dir, filename + ".pt")

        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            training_time = checkpoint['training_time']
            train_err = checkpoint['error']
            train_loss = checkpoint['loss']

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']+1))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            model, opt, training_time, train_err, train_loss, start_epoch = None, None, None, None, None, None

        return model, opt, training_time, train_err, train_loss, start_epoch

    def updateLogs(self, oldName, NewName, alg, ratio, epsilon, numIt, alpha, ratioADV, epochMax):
        import os
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./logs1", exist_ok=True)

        # Prefer ./logs, fallback ./logs1 (for older runs)
        candidates = [
            f"./logs/logs_{oldName}.txt",
            f"./logs1/logs_{oldName}.txt",
        ]
        src = next((p for p in candidates if os.path.isfile(p)), None)
        if src is None:
            print(f"[updateLogs] skip: missing logs for {oldName} (./logs or ./logs1)")
            return

        dst = f"./logs/logs_{NewName}.txt"

        with open(src, "r") as f_read, open(dst, "w") as f_write:
            for i, line in enumerate(f_read):
                if i == 0:
                    f_write.write(line if line.endswith("\n") else line + "\n")
                    continue

                raw = line.strip()
                if not raw:
                    continue

                aux = raw.split(",")
                try:
                    it = int(aux[0])
                except Exception:
                    # If this line is malformed, keep it as-is
                    f_write.write(raw + "\n")
                    continue

                if it > int(epochMax):
                    break

                # If the line already contains full fields (>=17), rewrite with new alg/params; else keep line
                if len(aux) >= 17:
                    algTest = aux[7]
                    eps_test = aux[8]
                    num_iterTest = aux[9]
                    alpha_test = aux[10]
                    adv_err = aux[11]
                    adv_loss = aux[12]
                    advTestTime = aux[13]
                    trainTime = aux[14]
                    test_entropy = aux[15]
                    test_MI = aux[16]

                    out = (
                        f"{it},{alg},{ratio},{epsilon},{numIt},{alpha},{ratioADV},"
                        f"{algTest},{eps_test},{num_iterTest},{alpha_test},"
                        f"{adv_err},{adv_loss},{advTestTime},{trainTime},{test_entropy},{test_MI}\n"
                    )
                    f_write.write(out)
                else:
                    f_write.write(raw + "\n")

 
    def standard_train(self, model, modelName, loader, dataset, opt, iterations=10, ckptName=None, runName=None):
        '''training a standard model with checkpoint and saving the model.''' 
        if ckptName is None: ckptName = modelName
        if runName  is None: runName  = modelName
        if self.deep_ensemble:
            return self.standard_train_deep_ensemble(model, modelName, loader, dataset, opt, iterations)
        
        if "binaryCifar10" in modelName: num_classes=2
        elif "cifar100" in modelName: num_classes=100
        elif "cifar10-c" in modelName: num_classes=10
        elif "cifar10" in modelName: num_classes=10
        elif "mnist" in modelName: num_classes=10
        elif "imageNet" in modelName: num_classes=1000
        else: num_classes=10

        lagrangian=None if not self.cals else AugLagrangianClass(num_classes=num_classes)
        if self.cals: print(lagrangian)

        write_pred_logs = True
        lr = opt.param_groups[0]['lr']
        savePrevModel = False
        num_samples = 5

        _model = None
        for it in range(iterations, 0, -1):
            model_name = ckptName + "_epoch" + str(it)
            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, model_name) # load model
            if _model is not None:
                # if models exists, load logs
                if ckptName != runName:
                    self.updateLogs(ckptName, runName, 'standard', 0, 0, 0, 0, 0, it)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                # ===== W&B anchor: step = last epoch of stage1 =====
                try:
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": float(train_loss),
                            "train/err":  float(train_err),
                            "epoch": int(it),
                            "stage": 1,
                            "anchor": 1,
                        }, step=int(it))
                        print(f"[W&B] anchor logged at step={it}", flush=True)
                except Exception as e:
                    print("[W&B anchor skipped]", e, flush=True)
                # =======================================================
                break
        print("Standard training")

        if _model is None: 
            it = 0
            counter = 0
            train_err = 0.0
            train_loss = 0.0
            t1 = time.time()

        for counter in range(it+1, iterations+1):
            if counter == iterations: write_pred_logs = True
            print("epoch number " + str(counter))

            self._ecg_on_epoch_begin(counter)
            self._current_epoch = int(counter)

            self._rt_train_begin()
            train_err, train_loss, misclassified_ids = self.epoch(loader.train_loader, model, opt, num_samples=num_samples, lagrangian=lagrangian)
            self._rt_train_end()

            self._rt_eval_begin()
            test_err, _, _, _, _, _, _, _ = self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)
            self._rt_eval_end()
            self._ecg_on_epoch_end(counter, metric=test_err)

            # ===== W&B: one point per epoch, step = epoch (1, 2, ..., 60) =====
            try:
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": float(train_loss),
                        "train/err":  float(train_err),
                        "epoch": int(counter),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "stage": 1,
                    }, step=int(counter))
                    print(f"[W&B] logged epoch={counter} loss={float(train_loss):.4f} err={float(train_err):.4f}", flush=True)
            except Exception as e:
                print("[W&B log skipped]", e, flush=True)
            # ==========================================

            # RunA/RunB: extra evals every eval_extra_every epochs (ADV/, C/, LT/ in W&B)
            self._run_extra_evals(counter, dataset, model, loader)

            if counter % 5 == 0 or counter == iterations:
                print("saving model on epoch " + str(counter))

                self.saveModel(True, {
                    'epoch': counter,
                    'state_dict': model.state_dict(),
                    'training_time': time.time() - t1,
                    'error': train_err,
                    'loss': train_loss,
                    'optimizer' : opt.state_dict(),}, ckptName, counter)


        print(f"[STAGE2] lr = {opt.param_groups[0]['lr']} (unchanged from stage1)", flush=True)

        write_pred_logs = True

        # ===== STAGE2 hard skip for sanity =====
        max_stage2_epochs = int(getattr(self, "stage2_epochs", 0))
        if max_stage2_epochs <= 0:
            print("[STAGE2] skipped because stage2_epochs=0", flush=True)
            return time.time() - t1, train_err, train_loss
        # ======================================

        # ===== ROBUST TRAINING STAGE2 (PGD-AT / TRADES / MART) =====
        _train_mode = getattr(self, "train_mode", "standard")
        if _train_mode in ("pgd_at", "trades", "mart"):
            print(f"[STAGE2-ROBUST] method={_train_mode}  eps={getattr(self, 'robust_eps', 0):.4f}"
                  f"  alpha={getattr(self, 'robust_alpha', 0):.4f}  steps={getattr(self, 'robust_steps', 10)}"
                  f"  beta={getattr(self, 'robust_beta', 6.0)}", flush=True)

            _rob_resume_ep = 0
            _model_dir = os.path.join(os.path.dirname(__file__), "models")
            for _chk_ep in range(max_stage2_epochs, 0, -1):
                _chk_global = _chk_ep + iterations
                _chk_path = os.path.join(_model_dir, f"{ckptName}_epoch{_chk_global}.pt")
                if os.path.isfile(_chk_path):
                    print(f"=> loading robust checkpoint '{_chk_path}'")
                    _chk = torch.load(_chk_path, map_location="cpu")
                    model.load_state_dict(_chk['state_dict'])
                    opt.load_state_dict(_chk['optimizer'])
                    t1 = time.time() - _chk['training_time']
                    _rob_resume_ep = _chk_ep
                    print(f"=> Resumed robust {_train_mode} from epoch {_chk_global} "
                          f"(stage2 ep {_chk_ep}/{max_stage2_epochs})")
                    break
            if _rob_resume_ep == 0:
                print("[STAGE2-ROBUST] No checkpoint found, starting from epoch 1")

            for ep in range(_rob_resume_ep + 1, max_stage2_epochs + 1):
                global_epoch = ep + iterations
                self._current_epoch = global_epoch
                self._ecg_on_epoch_begin(global_epoch)
                print(f"epoch number {global_epoch} (robust {_train_mode})")

                self._rt_train_begin()
                rob_err, rob_loss, _ = self.epoch_robust(
                    loader.train_loader, model, opt, method=_train_mode)
                self._rt_train_end()

                self._rt_eval_begin()
                test_err, _, _, _, _, _, _, _ = self.testModel_logs(
                    dataset, modelName, global_epoch, 'standard', 0, 0, 0, 0, 0,
                    time.time() - t1, False, num_samples=5)
                self._rt_eval_end()
                self._ecg_on_epoch_end(global_epoch, metric=test_err)

                try:
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": rob_loss, "train/err": rob_err,
                            "robust/eps": getattr(self, "robust_eps", 0),
                            "robust/method": _train_mode,
                            "epoch": global_epoch, "stage": 2,
                        }, step=global_epoch)
                except Exception:
                    pass

                if ep % 5 == 0 or ep == max_stage2_epochs:
                    self.saveModel(True, {
                        'epoch': global_epoch,
                        'state_dict': model.state_dict(),
                        'training_time': time.time() - t1,
                        'error': rob_err,
                        'loss': rob_loss,
                        'optimizer': opt.state_dict(),
                    }, ckptName, global_epoch)

                self._run_extra_evals(global_epoch, dataset, model, loader)

            return time.time() - t1, rob_err, rob_loss
        # ============================================================

        if option_stage2 == 'batch_mix2':
            #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
            # min error for correct batch
            # max unc for wrong batchs
            # interleve batches

            if Adaptive_Balancing: 
                alpha = torch.tensor(0.9)
                alpha_delta = 0.05
                test_err_prev, test_unc_prev = None, None

            elif dynamic_weights:
                # Define the loss weights as nn.Parameter objects
                weight_loss1 = torch.tensor(0.5, requires_grad=True)
                weight_loss2 = torch.tensor(0.5, requires_grad=True)
                weight_optimizer = optim.Adam([{'params':[weight_loss1, weight_loss2]}],lr=0.01)
                alpha=[weight_optimizer, weight_loss1, weight_loss2]

            else:
                alpha=None
                
            if savePrevModel: 
                prev_model = copy.deepcopy(model)
                prev_test_err = None
                prev_test_unc = None

            epoch_dataSize = len(loader.data_train)
            #epoch_counter = 0
            epoch_counter = 1

            counter_dataSize = 0
            counter_repeat = 0
            half_batch_size=int(loader.batch_size/2)

            max_stage2_epochs = int(self.new_iterations)
            # stage2_fast: reuse wrong/correct split every N epochs to cut full-train passes
            stage2_fast = getattr(self, "stage2_fast", False)
            stage2_find_every = getattr(self, "stage2_find_every", 3)
            stage2_ce_log_every = getattr(self, "stage2_ce_log_every", 5)
            _last_misclassified_ids = None
            _last_aux_correctclassified_ids = None
            _last_ce_loss = None
            _last_ce_err = None

            # ===== STAGE2 batch_mix2 checkpoint resume =====
            _model_dir = os.path.join(os.path.dirname(__file__), "models")
            for _chk_ep in range(max_stage2_epochs, 0, -1):
                _chk_global = _chk_ep + iterations
                _chk_path = os.path.join(_model_dir, f"{ckptName}_epoch{_chk_global}.pt")
                if os.path.isfile(_chk_path):
                    print(f"=> loading stage2 checkpoint '{_chk_path}'", flush=True)
                    _chk = torch.load(_chk_path, map_location="cpu")
                    model.load_state_dict(_chk['state_dict'])
                    opt.load_state_dict(_chk['optimizer'])
                    t1 = time.time() - _chk['training_time']
                    epoch_counter = _chk_ep + 1
                    print(f"=> Resumed stage2 (euat) from epoch {_chk_global} (stage2_ep {_chk_ep}/{max_stage2_epochs})", flush=True)
                    break
            # ===============================================

            print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))
            if stage2_fast:
                print(f"[STAGE2] fast mode: find wrong every {stage2_find_every} epochs, log train CE every {stage2_ce_log_every} epochs", flush=True)
            while epoch_counter < max_stage2_epochs+1:

                global_epoch = epoch_counter + iterations
                self._current_epoch = int(global_epoch)

                if counter_dataSize == 0 and counter_repeat == 0:

                    self._ecg_on_epoch_begin(global_epoch)
                if self.printTimes: t1_init = time.time()
                if counter_repeat==0:
                    # Run full-train "find misclassified" every stage2_find_every epochs when stage2_fast (epoch 1 always)
                    run_find = (not stage2_fast) or (epoch_counter == 1) or ((epoch_counter - 1) % stage2_find_every == 0)
                    if run_find or _last_misclassified_ids is None:
                        _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data
                        _last_misclassified_ids = misclassified_ids
                        aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
                        _last_aux_correctclassified_ids = aux_correctclassified_ids
                    else:
                        misclassified_ids = _last_misclassified_ids
                        aux_correctclassified_ids = _last_aux_correctclassified_ids
                    size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)

                    # update loss fucntion and train with wrong predicitons
                    _subset_wrong = Subset(loader.data_train, misclassified_ids)
                    _train_loader_wrong = DataLoader(_subset_wrong, batch_size=half_batch_size , shuffle=True) 

                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)
                _subset_correct = Subset(loader.data_train, correctclassified_ids)
                _train_loader_correct = DataLoader(_subset_correct, batch_size=half_batch_size, shuffle=True)
                #print("size correct " + str(len(_subset_correct)) + " size wrong " + str(len(_subset_wrong))) 
                if self.printTimes: print('time correct/wrong sets ' + str( time.time() - t1_init )) 

                # update loss fucntion and train with correct predicitons
                if self.printTimes: t1_init = time.time()
                self._rt_train_begin()
                self.epoch_interleave_batches(_train_loader_wrong,_train_loader_correct,  model, opt, num_samples=num_samples, weight_loss=alpha) # train
                self._rt_train_end()
                if self.printTimes: print('time train ' + str( time.time() - t1_init )) 
                counter_dataSize += len(misclassified_ids)+len(correctclassified_ids)

                counter_repeat += 1
                #if counter_repeat==10: counter_repeat=0
                if counter_repeat>0: counter_repeat=0

                #test
                if counter_dataSize > epoch_dataSize:
                    if self.printTimes: t1_init = time.time()

                    self._rt_eval_begin()
                    test_err, _, test_entropy, test_MI, _, _, _, _ = self.testModel_logs(dataset, runName, epoch_counter+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)
                    self._rt_eval_end()
                    test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI

                    last_global_epoch = iterations + max_stage2_epochs   # e.g. 30 + 30 = 60
                    # Run full-train compute_train_ce_err every stage2_ce_log_every epochs when stage2_fast (first & last stage2 epoch always)
                    run_ce_log = (not stage2_fast) or (epoch_counter == 1) or (global_epoch % stage2_ce_log_every == 0) or (global_epoch == last_global_epoch)
                    if run_ce_log:
                        ce_loss, ce_err = self.compute_train_ce_err(loader.train_loader, model)
                        _last_ce_loss, _last_ce_err = ce_loss, ce_err
                    else:
                        ce_loss = _last_ce_loss if _last_ce_loss is not None else 0.0
                        ce_err = _last_ce_err if _last_ce_err is not None else 0.0

                    # ===== SAVE CKPT in STAGE2 (every 5 global epochs + last) =====
                    if (global_epoch % 5 == 0) or (global_epoch == last_global_epoch):
                        print(f"[STAGE2] saving model on epoch {global_epoch}", flush=True)
                        self.saveModel(True, {
                            'epoch': global_epoch,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': float(test_err),
                            'loss':  float(ce_loss),
                            'optimizer': opt.state_dict(),
                        }, ckptName, global_epoch)
                    # =============================================================

                    # ---- log train metrics for stage2 ----
                    global_step = epoch_counter + iterations   # 31..60
                    self._ecg_on_epoch_end(global_step, metric=test_err)

                    try:
                        if wandb.run is not None:
                            wandb.log({
                                "train/loss": float(ce_loss),
                                "train/err":  float(ce_err),
                                "epoch": int(global_step),
                                "lr": float(opt.param_groups[0]["lr"]),
                                "stage": 2,
                            }, step=int(global_step))
                            print(f"[W&B] stage2 logged epoch={global_step} loss={ce_loss:.4f} err={ce_err:.4f}", flush=True)
                    except Exception as e:
                        print("[W&B log skipped]", e, flush=True)
                    # --------------------------------------

                    counter_dataSize = 0
                    epoch_counter +=1

                    if savePrevModel: 
                        if prev_test_err is None or (test_err < 2.0*prev_test_err):
                            prev_test_err = test_err
                            prev_test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                            prev_model = copy.deepcopy(model)
                        else:
                            print("Rolling out the model")
                            model = prev_model


                    if Adaptive_Balancing:
                        if test_err_prev is None:
                            test_err_prev = test_err
                            test_unc_prev = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI

                        if test_err > test_err_prev and test_unc > test_unc_prev:
                            # error and uncertainty increseas
                            # increase alpha to give more weight to min Err Correct and reduce the weight of max Unc wrong
                            alpha +=alpha_delta

                        elif test_err > test_err_prev and test_unc < test_unc_prev:
                            # error increseas and uncertainty decreseas
                            # increase alpha to give more weight to min Err Correct and reduce the weight of max Unc wrong
                            alpha +=alpha_delta

                        elif test_err < test_err_prev and test_unc > test_unc_prev:
                            # error decreseas and uncertainty increases
                            alpha -=alpha_delta
                        
                        #elif test_err < test_err_prev and test_unc < test_unc_prev:
                        #    # error and uncertainty decreseas
                        #    #keep alpha

                        test_err_prev = test_err
                        test_unc_prev = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                        alpha = torch.clamp(alpha, 0.0, 1.0)


                    print("epoch number " + str(epoch_counter+iterations))
                    if self.printTimes: print('time testing ' + str( time.time() - t1_init ))

                    # RunA/RunB: extra evals every eval_extra_every epochs in stage2
                    self._run_extra_evals(global_step, dataset, model, loader)

                #del _subset_wrong,_subset_correct, _train_loader_wrong, _train_loader_correct
                #torch.cuda.empty_cache()


        elif option_stage2 == 'batch_mix':
            #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
            # min error for correct batch
            # max unc for wrong batchs
            # first batches of wrong and then batches of correcrt
            for t in range(1, self.new_iterations+1):
                #if t == self.new_iterations: write_pred_logs = True
                print("epoch number " + str(t+iterations))

                _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data

                aux_correctclassified_ids = [i for i in range(len(loader.data_train)) if i not in misclassified_ids]
                size2include = len(misclassified_ids) if len(misclassified_ids) > loader.batch_size else loader.batch_size-len(misclassified_ids)
                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

                # update loss fucntion and train with wrong predicitons
                self.LossInUse = LOSS_2nd_stage_wrong
                _subset = Subset(loader.data_train, misclassified_ids)
                _train_loader = DataLoader(_subset, batch_size = loader.batch_size, shuffle=False) 
                self.epoch(_train_loader, model, opt) # train

                self.testModel_logs(dataset, modelName, t+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs=2, num_samples=num_samples)

                # update loss fucntion and train with correct predicitons
                self.LossInUse = LOSS_MIN_CROSSENT 
                _subset = Subset(loader.data_train, correctclassified_ids)
                _train_loader = DataLoader(_subset, batch_size = loader.batch_size, shuffle=False) 
                self.epoch(_train_loader, model, opt) # train

                #test
                self.testModel_logs(dataset, modelName, t+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)

                del _subset, _train_loader
                torch.cuda.empty_cache()


        elif option_stage2 == 'batch_granularity':
            epoch_dataSize = len(loader.data_train)
            #epoch_counter = 0
            epoch_counter = 1

            counter_dataSize = 0
            half_batch_size=int(loader.batch_size/2)

            print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))

            while epoch_counter < self.max_stage2_epochs+1:

                _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data
                aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
                size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)

                _subset_wrong = Subset(loader.data_train, misclassified_ids)
                _train_loader_wrong = DataLoader(_subset_wrong, batch_size=half_batch_size , shuffle=True) 

                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)
                _subset_correct = Subset(loader.data_train, correctclassified_ids)
                _train_loader_correct = DataLoader(_subset_correct, batch_size=half_batch_size, shuffle=True)

                # update loss fucntion and train with correct predicitons
                dataloader_iterator_wrong = iter(_train_loader_wrong)
                dataloader_iterator_correct = iter(_train_loader_correct)
        
                X_wrong,y_wrong = next(dataloader_iterator_wrong)
                X_correct,y_correct = next(dataloader_iterator_correct)

                X_wrong,y_wrong = X_wrong.to(self.device), y_wrong.to(self.device) # len of bacth size
                X_correct,y_correct = X_correct.to(self.device), y_correct.to(self.device) # len of bacth size

                if self.half_prec: 
                    # Runs the forward pass with autocasting.
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        y_pred_wrong = model(X_wrong)
                        y_pred_correct = model(X_correct)

                    if opt:   # backpropagation
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            self.LossInUse = LOSS_2nd_stage_wrong
                            loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=5)

                            self.LossInUse = LOSS_2nd_stage_correct
                            loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=5)

                            loss = loss_wrong + loss_correct

                        opt.zero_grad()
                        self.scaler.scale(loss).backward() # Scales the loss, and calls backward() . to create scaled gradients
                        self.scaler.step(opt) # Unscales gradients and calls or skips optimizer.step()
                        self.scaler.update()  # Updates the scale for next iteration


                else:
                    y_pred_wrong = model(X_wrong)
                    y_pred_correct = model(X_correct)

                    if opt:  # backpropagation
                        self.LossInUse = LOSS_2nd_stage_wrong
                        loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=5)

                        self.LossInUse = LOSS_2nd_stage_correct 
                        loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=5)

                        # Calculate the total loss with dynamic weights
                        loss = loss_wrong + loss_correct

                        opt.zero_grad()
                        loss.backward()
                        opt.step()



                counter_dataSize += half_batch_size*2.0 #len(misclassified_ids)+len(correctclassified_ids)
                #test
                if counter_dataSize > epoch_dataSize:
                    test_err, _, test_entropy, test_MI, _, _, _, _ = self.testModel_logs(
                        dataset, modelName,
                        counter,
                        'standard',
                        0, 0, 0, 0, 0,
                        time.time() - t1,
                        write_pred_logs,
                        num_samples=num_samples
                    )

                    counter_dataSize = 0

                    counter += 1
                    epoch_counter += 1

                    test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI

                    print(f"[STAGE2] finished epoch={counter-1}", flush=True)

                    #del _subset_wrong,_subset_correct, _train_loader_wrong, _train_loader_correct
                    #torch.cuda.empty_cache()


        else:
            self.LossInUse = LOSS_2nd_stage_wrong
            # mix correct and wrong prediciton in one batch and uses the same loss function
            for t in range(1, self.new_iterations+1):
                #if t == self.new_iterations: write_pred_logs = True
                print("epoch number " + str(t+iterations))

                _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data

                aux_correctclassified_ids = [i for i in range(len(loader.data_train)) if i not in misclassified_ids]
                size2include = len(misclassified_ids) if len(misclassified_ids) > loader.batch_size else loader.batch_size-len(misclassified_ids)
                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

                _subset = Subset(loader.data_train, misclassified_ids+correctclassified_ids)
                _train_loader = DataLoader(_subset, batch_size = loader.batch_size, shuffle=False) 
                _, _, misclassified_ids = self.epoch(_train_loader, model, opt) # train

                self.testModel_logs(dataset, modelName, t+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)

                del _subset, _train_loader
                torch.cuda.empty_cache()



        if self.ToCalibrate:
            print("Model Calibration")
            self.calibrate(loader.data_val, model, num_samples=1, )
            self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=True)


        elif self.deup:
            print("DEUP")
            
            self.deup_model = DEUP(loader, model, opt, self.device)
            self.deup_model.train()

            self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=False)

        trainTime = time.time() - t1

        return (trainTime, train_err, train_loss)


    def standard_pgd_train(self, model, modelName, loader, dataset, opt, iterations=10, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        
        if self.deep_ensemble:
            return self.standard_train_deep_ensemble(model, modelName, loader, dataset, opt, iterations,ckptName=ckptName, runName=runName)

        num_samples = 5

        #loading the model
        write_pred_logs = True

        #update hyper-parameter for adversarial training
        lr = opt.param_groups[0]['lr']
        momentum = opt.param_groups[0]["momentum"]

        # load final models
        _model = None
        for it in range(iterations, 0, -1):
            model_name = modelName + "_epoch" + str(it) #str(iterations)
            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, model_name)
            if _model is not None:
                self.updateLogs(runName, runName, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, it) #iterations)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                break


        if _model is None: 
            # but check if we have a clean data model first
            auxName = modelName.split("_")
            ratioStd = 1 - ratio
            it = int(iterations * ratioStd)

            old_name = auxName[0] + "_std_train_" + auxName[8] + "_" + auxName[9] + "_" + auxName[10] + "_lrAdv0.0_momentumAdv0.0_batchAdv0"
            old_model_name = old_name + "_epoch" + str(it)

            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, old_model_name)
            if _model is not None:
                self.updateLogs(old_name, modelName, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, it)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1

        if _model is None: 
            t1 = time.time()
            counter = 1
            train_err = 0.0
            train_loss = 0.0

        ST_it = int(iterations*(1-ratio))
        AT_it = int(iterations*ratio)

        for _ in range(counter, ST_it+1):
            print("epoch number (ST) " + str(counter))
            train_err, train_loss, _ = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs)

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))

                self.saveModel(True, {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': train_err,
                            'loss': train_loss,
                            'optimizer' : opt.state_dict(),}, old_name, counter-1)
            counter += 1


        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv


        for t in range(counter, AT_it+1):
            print("epoch number  (AT)" + str(counter))
            #if counter == iterations: write_pred_logs=True

            train_err, train_loss, misclassified_ids_adv = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, \
                                                                                        alpha=alpha_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset,modelName,counter,'std_pgd',ratio,eps_train,num_iterTrain,alpha_train,ratio_adv,time.time()-t1,write_pred_logs)
            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))
                self.saveModel(True, {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': train_err,
                            'loss': train_loss,
                            'optimizer' : opt.state_dict(),}, modelName, counter)

            counter += 1

            
        #
        # here we end the normal training (ST + AT)
        #


        for param_group in opt.param_groups:
            param_group["lr"] = lr if not cycle_lr else  10e-4 #lr/10.0
            param_group["momentum"] = momentum                  


        #write_pred_logs = True
        #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
        # min error+unc for correct batch
        # min error + max unc for wrong batchs
        # interleve batches
        epoch_dataSize = len(loader.data_train)
        epoch_counter = 1
        counter_dataSize = 0
        half_batch_size=int(loader.batch_size/2)

        itST = int(self.new_iterations*(1-ratio)) 
        itAT = int(self.new_iterations*ratio)

        print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))
        while epoch_counter < itST+1:
            _, _, misclassified_ids = self.epoch(loader.train_loader, model) # to determine wrong inputs

            aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
            size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)
            correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = half_batch_size, shuffle=False) 

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = half_batch_size, shuffle=False) 

            self.epoch_interleave_batches(_train_loader_wrong,_train_loader_correct,  model, opt) # train
            counter_dataSize += len(misclassified_ids)+len(correctclassified_ids)


            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_correct, _train_loader_correct, _subset_wrong, _train_loader_wrong
            torch.cuda.empty_cache()

        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv if not cycle_lr else 10e-4 # lr_adv/10.0
            param_group["momentum"] = momentum_adv
        half_batch_size_adv=int(loader.batch_size_adv/2)
        epoch_counter = 1
        counter_dataSize = 0

        while epoch_counter < itAT+1:
            
            _, _, misclassified_ids_adv = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, \
                                                                                        alpha=alpha_train, ratio=ratio_adv, opt=None)

            aux_correctclassified_ids_adv = [i for i in range(epoch_dataSize) if i not in misclassified_ids_adv]
            size2include = len(misclassified_ids_adv) if len(misclassified_ids_adv) > half_batch_size_adv else loader.batch_size_adv-len(misclassified_ids_adv)
            correctclassified_ids_adv = random.choices(aux_correctclassified_ids_adv, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids_adv)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = loader.batch_size_adv, shuffle=False)

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids_adv)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = loader.batch_size_adv, shuffle=False)
            
            self.epoch_adversarial_interleave_batches(_train_loader_wrong,_train_loader_correct,model,"pgd",\
                                                        dataset,epsilon=eps_train,num_iter=num_iterTrain,alpha=alpha_train,ratio=ratio_adv,opt=opt)
            counter_dataSize += len(misclassified_ids_adv)+len(correctclassified_ids_adv)

            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_wrong,_subset_correct,_train_loader_wrong, _train_loader_correct
            torch.cuda.empty_cache()


        if self.ToCalibrate:
            print("Model Calibration")
            self.calibrate_adversarial(loader.data_val, model, num_samples=1, attack="pgd", epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio ,eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=True)

        elif self.deup:
            print("DEUP")
            
            self.deup_model = DEUP(loader, model, opt, self.device)
            self.deup_model.train(algorithm='pgd', epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train)

            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio ,eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=False)


        trainTime = time.time() - t1

        return (trainTime, train_err, train_loss)


    def standard_fgsm_train(self, model, modelName, loader, dataset, opt, iterations=10, ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        if self.deep_ensemble:
            #return self.standard_train_deep_ensemble(model, modelName, loader, dataset, opt, iterations)
            return self.standard_fgsm_train_deep_ensemble(model, modelName, loader, dataset, opt, iterations, epsilon=eps_train, ratio=ratio, ratio_adv=ratio_adv, lr_adv=lr_adv, momentum_adv=momentum_adv)
                  
        num_samples = 5
       #loading the model
        write_pred_logs = True

        #update hyper-parameter for adversarial training
        lr = opt.param_groups[0]['lr']
        momentum = opt.param_groups[0]["momentum"]

        # load final models
        _model = None
        for it in range(iterations, 0, -1):
            model_name = modelName + "_epoch" + str(it) #+ str(iterations)
            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, model_name)
            if _model is not None:
                self.updateLogs(modelName, modelName, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, iterations)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                break


        if _model is None: 
            # but check if we have a clean data model first
            auxName = modelName.split("_")
            ratioStd = 1.0 - ratio
            epoch = int(iterations * ratioStd)

            old_name = auxName[0] + "_std_train_" + auxName[6] + "_" + auxName[7] + "_" + auxName[8] + "_lrAdv0.0_momentumAdv0.0_batchAdv0"
            old_model_name = old_name + "_epoch" + str(epoch)

            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, old_model_name)
            if _model is not None:
                self.updateLogs(old_name, modelName, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, epoch)
                t1 = time.time()-trainTime
                opt = _opt
                model = _model
                counter += 1
         
        if _model is None: 
            t1 = time.time()
            counter = 1
            train_err = 0.0
            train_loss = 0.0

        ST_it = int(iterations*(1-ratio))
        AT_it = int(iterations*ratio)


        for t in range(counter, ST_it+1):
            print("epoch number (ST) " + str(counter))
            train_err, train_loss, _ = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))

                self.saveModel(True, {
                    'epoch': counter,
                    'state_dict': model.state_dict(),
                    'training_time': time.time() - t1,
                    'error': train_err,
                    'loss': train_loss,
                    'optimizer' : opt.state_dict(),}, old_name, counter)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for _ in range(counter, AT_it+1):
            #if counter == iterations: write_pred_logs = True
            print("epoch number (AT) " + str(counter))
            train_err, train_loss, _ = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset,modelName,counter,'std_fgsm',ratio,eps_train,0,0,ratio_adv,time.time()-t1,write_pred_logs)  

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))
                self.saveModel(True, {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': train_err,
                            'loss': train_loss,
                            'optimizer' : opt.state_dict(),}, modelName, counter)
            counter += 1

        #
        # here we end the normal training (ST + AT)
        #

        for param_group in opt.param_groups: # restore the ST HP values
            param_group["lr"] = lr if not cycle_lr else 10e-4 #  lr/10.0
            param_group["momentum"] = momentum     


        #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
        # min error+unc for correct batch
        # min error + max unc for wrong batchs
        # interleve batches
        epoch_dataSize = len(loader.data_train)
        epoch_counter = 1
        counter_dataSize = 0
        half_batch_size=int(loader.batch_size/2)

        itST = int(self.new_iterations*(1-ratio)) 
        itAT = int(self.new_iterations*ratio)

        print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))
        while epoch_counter < itST+1:
            _, _, misclassified_ids = self.epoch(loader.train_loader, model) # to determine wrong inputs

            aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
            size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)
            correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = half_batch_size, shuffle=False) 

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = half_batch_size, shuffle=False) 

            self.epoch_interleave_batches(_train_loader_wrong,_train_loader_correct,  model, opt) # train
            counter_dataSize += len(misclassified_ids)+len(correctclassified_ids)

            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_correct, _train_loader_correct, _subset_wrong, _train_loader_wrong
            torch.cuda.empty_cache()

        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv if not cycle_lr else 10e-4 # lr_adv/10.0
            param_group["momentum"] = momentum_adv
        half_batch_size_adv=int(loader.batch_size_adv/2)
        epoch_counter = 1
        counter_dataSize = 0

        while epoch_counter < itAT+1:
            
            _, _, misclassified_ids_adv = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv)

            aux_correctclassified_ids_adv = [i for i in range(epoch_dataSize) if i not in misclassified_ids_adv]
            size2include = len(misclassified_ids_adv) if len(misclassified_ids_adv) > half_batch_size_adv else loader.batch_size_adv-len(misclassified_ids_adv)
            correctclassified_ids_adv = random.choices(aux_correctclassified_ids_adv, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids_adv)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = loader.batch_size_adv, shuffle=False)

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids_adv)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = loader.batch_size_adv, shuffle=False)
            
            self.epoch_adversarial_interleave_batches(_train_loader_wrong,_train_loader_correct,model,"fgsm",dataset,epsilon=eps_train,ratio=ratio_adv,opt=opt)
            counter_dataSize += len(misclassified_ids_adv)+len(correctclassified_ids_adv)

            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_wrong,_subset_correct,_train_loader_wrong, _train_loader_correct
            torch.cuda.empty_cache()


        if self.ToCalibrate:
            print("Model Calibration")
            self.calibrate_adversarial(loader.data_val, model, num_samples=1, attack="fgsm", epsilon=eps_train)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio ,eps_train, 0, 0, ratio_adv, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=True)

        elif self.deup:
            print("DEUP")
            self.deup_model = DEUP(loader, model, opt, self.device)
            self.deup_model.train(algorithm='fgsm', epsilon=eps_train)
            
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio ,eps_train, 0, 0, ratio_adv, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=False)



        trainTime = time.time() - t1

        return (trainTime, train_err, train_loss)


    def standard_train_deep_ensemble(self, model, modelName, loader, dataset, opt, iterations=10):
        t1 = time.time()
        write_pred_logs = True
        num_samples = 1

        cuda = False if 'cpu' in str(self.device) else True

        _model = copy.deepcopy(model)

        self.model = FusionClassifier(
            estimator=_model,
            n_estimators=3,
            cuda=cuda,
            #save_model=False
        )

        criterion = nn.CrossEntropyLoss()
        self.model.set_criterion(criterion)

        for param_group in opt.param_groups:
            lr = param_group['lr']
            momentum = param_group['momentum']
            dampening = param_group['dampening']
            weight_decay = param_group['weight_decay']
            nesterov = param_group['nesterov']
        
        #self.model.set_optimizer('SGD',lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening, nesterov=nesterov) 
        self.model.set_optimizer('SGD',lr=0.1, weight_decay=10e-5, momentum=0.9)

        train_loader = DataLoader(loader.data_train, batch_size = 64, shuffle=False)
        #self.model.fit(train_loader=loader.train_loader,epochs=iterations,  obj_test=self, dataset=dataset, modelName=modelName, write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1)  
        self.model.fit(train_loader=train_loader,epochs=iterations,  obj_test=self, dataset=dataset, modelName=modelName, write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1)  

        return (0, 0, 0)


    def standard_fgsm_train_deep_ensemble(self, model, modelName, loader, dataset, opt, iterations=10, epsilon=0.1, ratio=1.0, ratio_adv=1.0, lr_adv=0.1, momentum_adv=0.9):
        version = 0
        t1 = time.time()
        write_pred_logs = True
        num_samples = 1

        cuda = False if 'cpu' in str(self.device) else True

        _model = copy.deepcopy(model)

        if version == 0:
            self.model = AdversarialTrainingClassifier(
                estimator=_model,
                n_estimators=3,
                cuda=cuda,
                #save_model=False
            )

            criterion = nn.CrossEntropyLoss()
            self.model.set_criterion(criterion)

            for param_group in opt.param_groups:
                #lr = param_group['lr']
                #momentum = param_group['momentum']
                dampening = param_group['dampening']
                weight_decay = param_group['weight_decay']
                nesterov = param_group['nesterov']
            
            self.model.set_optimizer('SGD',lr=lr_adv, weight_decay=weight_decay, momentum=momentum_adv, dampening=dampening, nesterov=nesterov) 

            self.model.fit(train_loader=loader.trainAvd_loader, epochs=iterations, save_model=False, obj_test=self, dataset=dataset, modelName=modelName, \
                                write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1, epsilon=epsilon, algorithm='fgsm', num_iter=0, alpha=0, \
                                ratio=ratio, ratio_adv=ratio_adv)  

        else:
            self.model = FusionClassifier(
                estimator=_model,
                n_estimators=3,
                cuda=cuda,
                #save_model=False
            )

            criterion = nn.CrossEntropyLoss()
            self.model.set_criterion(criterion)

            for param_group in opt.param_groups:
                #lr = param_group['lr']
                #momentum = param_group['momentum']
                dampening = param_group['dampening']
                weight_decay = param_group['weight_decay']
                nesterov = param_group['nesterov']
            
            self.model.set_optimizer('SGD',lr=lr_adv, weight_decay=weight_decay, momentum=momentum_adv, dampening=dampening, nesterov=nesterov) 

            self.model.fit_adversarial_train(train_loader=loader.trainAvd_loader, epochs=iterations, save_model=False, test_loader=loader.test_loader, \
                                             obj_test=self, dataset=dataset, modelName=modelName, write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1,\
                                             algorithm='fgsm',epsilon=epsilon, ratio=ratio, ratio_adv=ratio_adv)  




        return (0, 0, 0)


    def standard_pgd_train_deep_ensemble(self, model, modelName, loader, dataset, opt, iterations=10, epsilon=0.1, num_iter=20, alpha=0.01, ratio=1.0, ratio_adv=1.0, lr_adv=0.1, momentum_adv=0.9):
        version = 0
        t1 = time.time()
        write_pred_logs = True
        num_samples = 1

        cuda = False if 'cpu' in str(self.device) else True

        _model = copy.deepcopy(model)

        if version == 0:
            self.model = AdversarialTrainingClassifier(
                estimator=_model,
                n_estimators=3,
                cuda=cuda,
                #save_model=False
            )

            criterion = nn.CrossEntropyLoss()
            self.model.set_criterion(criterion)

            for param_group in opt.param_groups:
                #lr = param_group['lr']
                #momentum = param_group['momentum']
                dampening = param_group['dampening']
                weight_decay = param_group['weight_decay']
                nesterov = param_group['nesterov']
            
            self.model.set_optimizer('SGD',lr=lr_adv, weight_decay=weight_decay, momentum=momentum_adv, dampening=dampening, nesterov=nesterov) 

            self.model.fit(train_loader=loader.trainAvd_loader, epochs=iterations, save_model=False, obj_test=self, dataset=dataset, modelName=modelName, \
                                write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1, epsilon=epsilon, algorithm='pgd', num_iter=num_iter, alpha=alpha, \
                                ratio=ratio, ratio_adv=ratio_adv)  

        else:

            self.model = FusionClassifier(
                estimator=_model,
                n_estimators=3,
                cuda=cuda,
                #save_model=False
            )

            criterion = nn.CrossEntropyLoss()
            self.model.set_criterion(criterion)

            for param_group in opt.param_groups:
                #lr = param_group['lr']
                #momentum = param_group['momentum']
                dampening = param_group['dampening']
                weight_decay = param_group['weight_decay']
                nesterov = param_group['nesterov']
            
            self.model.set_optimizer('SGD',lr=lr_adv, weight_decay=weight_decay, momentum=momentum_adv, dampening=dampening, nesterov=nesterov) 

            self.model.fit_adversarial_train(train_loader=loader.trainAvd_loader,epochs=iterations, save_model=False, test_loader=loader.test_loader, \
                                            obj_test=self, dataset=dataset, modelName=modelName, write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1,\
                                            algorithm='pgd',epsilon=epsilon, num_iter=num_iter, alpha=alpha, ratio=ratio, ratio_adv=ratio_adv)  

        return (0, 0, 0)


    def LossFunction(self, model, X, y, y_pred, num_samples=5, CrossEntropyFunction=False):
        # LOSS_MIN_CROSSENT = 0 # minimize cross entropy loss
        # LOSS_MIN_CROSSENT_UNC  = 1 # minimize cross_entropy_loss + uncertainty
        # LOSS_MIN_CROSSENT_MAX_UNC = 2 # minimize cross_entropy_loss - uncertainty = minimize cross_entropy_loss + maximize uncertainty
        # LOSS_MIN_UNC = 3 # minimize -uncertainty = maximize uncertainty
    
        # ---- runtime diag: time the loss call (sampled) ----
        rt_token = None
        try:
            bs = int(y_pred.shape[0]) if hasattr(y_pred, "shape") and len(y_pred.shape) > 0 else 0
        except Exception:
            bs = 0
        try:
            rt_token = self._rt_loss_timer_start(batch_size=bs)
        except Exception:
            rt_token = None
    
        def _rt_ret(val):
            try:
                self._rt_loss_timer_end(rt_token)
            except Exception:
                pass
            return val
    
        if self.LossInUse == LOSS_ECG:
            # Auto-lambda: lam from gate_ema and delta (no schedule). auto_w: delta_eff = delta * min(1, epoch/5).
            # auto_d/auto_dw: delta_eff from _ecg_delta_eff (updated at epoch end); auto_dw = 5-epoch warmup on delta_eff.
            ecg_lam_rule = getattr(self, "ecg_lam_rule", None)
            use_lam_auto = ecg_lam_rule in _AUTO_LAM_RULES
            if use_lam_auto:
                gate_ema = getattr(self, "_ecg_gate_ema", None)
                delta = getattr(self, "ecg_lam_delta", 0.05)
                eps = getattr(self, "ecg_lam_eps", 1e-6)
                cur_epoch = float(getattr(self, "_current_epoch", 1))
                lam_max = float(getattr(self, "ecg_lam_max", 1.5))  # fixed
                if ecg_lam_rule == "auto_w":
                    warmup_epochs = 5
                    delta_eff = delta * min(1.0, cur_epoch / warmup_epochs)
                elif ecg_lam_rule == "auto_d":
                    delta_eff = getattr(self, "_ecg_delta_eff", delta)
                elif ecg_lam_rule == "auto_dw":
                    warmup_epochs = getattr(self, "ecg_auto_d_warmup_epochs", 5)
                    delta_eff = getattr(self, "_ecg_delta_eff", delta) * min(1.0, cur_epoch / warmup_epochs)
                elif ecg_lam_rule in ("auto_tr", "auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                    mean_g = getattr(self, "_ecg_gate_ema", None)
                    hi_g = getattr(self, "_ecg_gate_p99_ema", None)
                    r = float(getattr(self, "ecg_tail_ratio_target", 3.0))
                    prev_smooth = getattr(self, "_ecg_lam_auto_tr_ema", None)
                    if mean_g is None or hi_g is None:
                        lam_target = 0.0
                        self._ecg_lam_auto_raw = 0.0
                    else:
                        denom = hi_g - r * mean_g
                        if denom <= 1e-8:
                            inv_decay = float(getattr(self, "ecg_tail_invalid_decay", 0.95))
                            lam_target = (prev_smooth * inv_decay) if prev_smooth is not None else 0.0
                        else:
                            lam_target = (r - 1.0) / denom
                        lam_target = min(lam_max, max(0.0, lam_target))
                        self._ecg_lam_auto_raw = lam_target
                    lam_beta = float(getattr(self, "ecg_tail_lam_ema", 0.9))
                    if prev_smooth is None or lam_beta <= 0.0:
                        lam_smooth = lam_target
                    else:
                        lam_smooth = lam_beta * prev_smooth + (1.0 - lam_beta) * lam_target
                    lam_smooth = min(lam_max, max(0.0, lam_smooth))
                    self._ecg_lam_auto_tr_ema = lam_smooth
                    self._ecg_lam_auto_smoothed = lam_smooth
                    lam_base = lam_smooth
                    # auto_tr_sustain / auto_tr_autocap: additive sustain floor
                    self._ecg_lam_sustain_target = 0.0
                    if ecg_lam_rule in ("auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                        _SUSTAIN_TARGET_FLOOR = 1.08
                        _SUSTAIN_FRAC = 0.25
                        _SUSTAIN_BETA = 0.95
                        s_ema = getattr(self, "_ecg_scale_p99_ema_tr", None)
                        afrac_s = getattr(self, "_ecg_active_frac_ema", None)
                        floor_s = float(getattr(self, "ecg_active_frac_floor", 0.05))
                        gate_healthy = (afrac_s is not None and afrac_s >= floor_s)
                        if s_ema is not None and gate_healthy and s_ema < _SUSTAIN_TARGET_FLOOR:
                            deficit = min(1.0, max(0.0, (_SUSTAIN_TARGET_FLOOR - s_ema) / max(_SUSTAIN_TARGET_FLOOR - 1.0, 1e-8)))
                            sustain_tgt = lam_max * _SUSTAIN_FRAC * deficit
                        else:
                            sustain_tgt = 0.0
                        self._ecg_lam_sustain_target = sustain_tgt
                        prev_sustain = getattr(self, "_ecg_lam_sustain_ema", 0.0)
                        sustain_val = _SUSTAIN_BETA * prev_sustain + (1.0 - _SUSTAIN_BETA) * sustain_tgt
                        self._ecg_lam_sustain_ema = sustain_val
                        lam_inner = min(lam_max, max(lam_base, sustain_val))
                    else:
                        lam_inner = lam_base
                    # auto_tr_autocap: dynamic runtime cap
                    self._ecg_cap_up_pressure = 0.0
                    self._ecg_cap_down_pressure = 0.0
                    if ecg_lam_rule in ("auto_tr_autocap", "auto_tr_autocap_gate"):
                        _CAP_SCALE_LOW = 1.08
                        _CAP_SCALE_HIGH = 1.18
                        _CAP_UP_RATE = 0.05
                        _CAP_DOWN_RATE = 0.08
                        _CAP_HIT_BETA = 0.95
                        cap_cur = getattr(self, "_ecg_lam_cap_cur", min(lam_max, 1.0))
                        cap_min = getattr(self, "_ecg_lam_cap_min", min(0.8, lam_max))
                        cap_hit = 1.0 if lam_inner >= 0.98 * cap_cur else 0.0
                        prev_hit_ema = getattr(self, "_ecg_lam_cap_hit_ema", 0.0)
                        self._ecg_lam_cap_hit_ema = _CAP_HIT_BETA * prev_hit_ema + (1.0 - _CAP_HIT_BETA) * cap_hit
                        s_ema_c = getattr(self, "_ecg_scale_p99_ema_tr", None)
                        afrac_c = getattr(self, "_ecg_active_frac_ema", None)
                        floor_c = float(getattr(self, "ecg_active_frac_floor", 0.05))
                        up_pressure = 0.0
                        down_pressure = 0.0
                        if s_ema_c is not None:
                            if s_ema_c > _CAP_SCALE_HIGH:
                                scale_excess = min(1.0, max(0.0, (s_ema_c - _CAP_SCALE_HIGH) / max(_CAP_SCALE_HIGH - 1.0, 1e-8)))
                                down_pressure = max(down_pressure, scale_excess)
                            if s_ema_c < _CAP_SCALE_LOW and self._ecg_lam_cap_hit_ema > 0.3 and afrac_c is not None and afrac_c >= floor_c:
                                scale_deficit = min(1.0, max(0.0, (_CAP_SCALE_LOW - s_ema_c) / max(_CAP_SCALE_LOW - 1.0, 1e-8)))
                                up_pressure = scale_deficit * self._ecg_lam_cap_hit_ema
                        if afrac_c is not None and afrac_c < floor_c:
                            active_bad = min(1.0, max(0.0, (floor_c - afrac_c) / max(floor_c, 1e-8)))
                            down_pressure = max(down_pressure, active_bad)
                        self._ecg_cap_up_pressure = up_pressure
                        self._ecg_cap_down_pressure = down_pressure
                        cap_cur += _CAP_UP_RATE * (lam_max - cap_cur) * up_pressure
                        cap_cur -= _CAP_DOWN_RATE * (cap_cur - cap_min) * down_pressure
                        cap_cur = min(lam_max, max(cap_min, cap_cur))
                        self._ecg_lam_cap_cur = cap_cur
                        self._ecg_lam_inner_before_cap = lam_inner
                        lam_cur = min(lam_inner, cap_cur)
                    else:
                        lam_cur = lam_inner
                    afrac = getattr(self, "_ecg_active_frac_ema", None)
                    floor = float(getattr(self, "ecg_active_frac_floor", 0.05))
                    if afrac is not None and afrac < floor:
                        if getattr(self, "ecg_sparse_lam_zero", False):
                            lam_cur = 0.0
                        else:
                            lam_cur *= float(getattr(self, "ecg_sparse_lam_decay", 0.5))
                    self._ecg_lam_auto_after_guard = lam_cur
                else:
                    delta_eff = delta
                if ecg_lam_rule not in ("auto_tr", "auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                    if gate_ema is None:
                        lam_cur = lam_max
                    else:
                        lam_cur = min(lam_max, max(0.0, delta_eff / (gate_ema + eps)))
            else:
                lam_cur = self.ecg_lam
            ecg_tau_rule = getattr(self, "ecg_tau_rule", None)
            if ecg_tau_rule == "quantile":
                tau_q = getattr(self, "ecg_tau_quantile", 0.8)
                loss, stats = ecg_loss(
                    y_pred, y,
                    lam=lam_cur,
                    tau=self.ecg_tau,
                    k=self.ecg_k,
                    conf_type=self.ecg_conf_type,
                    detach_gates=getattr(self, "ecg_detach_gates", True),
                    tau_quantile=tau_q,
                    scale_normalize=use_lam_auto,
                    gate_temp=getattr(self, "ecg_gate_temp", 1.5),
                )
            elif ecg_tau_rule in ("auto_q", "auto_q_ctrl", "auto_q_valley"):
                tau_q = getattr(self, "ecg_tau_quantile_cur", getattr(self, "ecg_tau_q_start", 0.6))
                # auto_tr_autocap_gate: weak anti-over-narrow correction
                self._ecg_gate_q_base = tau_q
                self._ecg_gate_narrow_deficit = 0.0
                if ecg_lam_rule == "auto_tr_autocap_gate":
                    _GATE_FLOOR = 0.12
                    _GATE_Q_MAX_CORR = 0.15
                    afrac_g = getattr(self, "_ecg_active_frac_ema", None)
                    if afrac_g is not None and afrac_g < _GATE_FLOOR:
                        deficit = min(1.0, max(0.0, (_GATE_FLOOR - afrac_g) / max(_GATE_FLOOR, 1e-8)))
                        q_corr = _GATE_Q_MAX_CORR * deficit
                        tau_q = max(0.1, tau_q - q_corr)
                        self._ecg_gate_narrow_deficit = deficit
                        self._ecg_gate_q_correction = q_corr
                    else:
                        self._ecg_gate_q_correction = 0.0
                loss, stats = ecg_loss(
                    y_pred, y,
                    lam=lam_cur,
                    tau=self.ecg_tau,
                    k=self.ecg_k,
                    conf_type=self.ecg_conf_type,
                    detach_gates=getattr(self, "ecg_detach_gates", True),
                    tau_quantile=tau_q,
                    scale_normalize=use_lam_auto,
                    gate_temp=getattr(self, "ecg_gate_temp", 1.5),
                )
            else:
                loss, stats = ecg_loss(
                    y_pred, y,
                    lam=lam_cur,
                    tau=self.ecg_tau,
                    k=self.ecg_k,
                    conf_type=self.ecg_conf_type,
                    detach_gates=getattr(self, "ecg_detach_gates", True),
                    scale_normalize=use_lam_auto,
                    gate_temp=getattr(self, "ecg_gate_temp", 1.5),
                )
            # Auto-lambda: update gate_mean_ema (_ecg_gate_ema) for all auto rules including auto_tr;
            # lam_cur_raw and tail_ratio_est both depend on it. Optional DDP sync.
            if use_lam_auto and model.training:
                gate_mean = stats.get("gate_mean", 0.0)
                try:
                    if torch.distributed.is_initialized():
                        t = torch.tensor([gate_mean], dtype=torch.float32, device=y_pred.device)
                        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                        gate_mean = t.item()
                except Exception:
                    pass
                beta = getattr(self, "ecg_lam_beta", 0.9)
                if getattr(self, "_ecg_gate_ema", None) is None:
                    self._ecg_gate_ema = gate_mean
                else:
                    self._ecg_gate_ema = beta * self._ecg_gate_ema + (1.0 - beta) * gate_mean
                self.ecg_lam = lam_cur  # for epoch-level wandb log
                if ecg_lam_rule in ("auto_tr", "auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                    beta_tr = getattr(self, "ecg_tail_ratio_beta", 0.9)
                    gate_p95 = stats.get("gate_p95", 0.0)
                    gate_p99 = stats.get("gate_p99", 0.0)
                    afrac = stats.get("conf_gate_active_frac", 0.0)
                    if getattr(self, "_ecg_gate_p95_ema", None) is None:
                        self._ecg_gate_p95_ema = gate_p95
                    else:
                        self._ecg_gate_p95_ema = beta_tr * self._ecg_gate_p95_ema + (1.0 - beta_tr) * gate_p95
                    if getattr(self, "_ecg_gate_p99_ema", None) is None:
                        self._ecg_gate_p99_ema = gate_p99
                    else:
                        self._ecg_gate_p99_ema = beta_tr * self._ecg_gate_p99_ema + (1.0 - beta_tr) * gate_p99
                    if getattr(self, "_ecg_active_frac_ema", None) is None:
                        self._ecg_active_frac_ema = afrac
                    else:
                        self._ecg_active_frac_ema = beta_tr * self._ecg_active_frac_ema + (1.0 - beta_tr) * afrac
                    if ecg_lam_rule in ("auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                        sp99 = stats.get("scale_p99_after_norm", 0.0)
                        if getattr(self, "_ecg_scale_p99_ema_tr", None) is None:
                            self._ecg_scale_p99_ema_tr = sp99
                        else:
                            self._ecg_scale_p99_ema_tr = beta_tr * self._ecg_scale_p99_ema_tr + (1.0 - beta_tr) * sp99
            if getattr(self, "use_wandb", False):
                # Accumulate ECG gate stats (avoid per-batch wandb spam)
                try:
                    if not hasattr(self, "_ecg_stat_sum"):
                        self._ecg_stat_sum = {}
                        self._ecg_stat_n = 0
                    self._ecg_stat_n += 1
                    for _k, _v in stats.items():
                        if _k == "_conf_hist":
                            # accumulate histogram bin counts as list (not averaged with _ecg_stat_n)
                            try:
                                _hist = list(_v)
                                if "_conf_hist" not in self._ecg_stat_sum:
                                    self._ecg_stat_sum["_conf_hist"] = [0.0] * len(_hist)
                                self._ecg_stat_sum["_conf_hist"] = [
                                    a + b for a, b in zip(self._ecg_stat_sum["_conf_hist"], _hist)
                                ]
                            except Exception:
                                pass
                            continue
                        try:
                            _fv = float(_v)
                        except Exception:
                            # torch tensors
                            try:
                                _fv = float(_v.detach().cpu().item())
                            except Exception:
                                continue
                        self._ecg_stat_sum[_k] = self._ecg_stat_sum.get(_k, 0.0) + _fv
                except Exception:
                    pass
    
            return _rt_ret(loss)
    
        if CrossEntropyFunction:
            # to test the model
            return _rt_ret(nn.CrossEntropyLoss()(y_pred, y))  # CrossEntropy_Loss()(model, X, y, y_pred, num_samples)
    
        if self.LossInUse == LOSS_MIN_CROSSENT_UNC:
            return _rt_ret(CrossEntropy_Uncertainty_Loss()(model, X, y, y_pred, num_samples))
        elif self.LossInUse == LOSS_MIN_CROSSENT_MAX_UNC:
            return _rt_ret(CrossEntropy_Certainty_Loss()(model, X, y, y_pred, num_samples))
        elif self.LossInUse == LOSS_MIN_UNC:
            return _rt_ret(Uncertainty_Loss()(model, X, y, y_pred, num_samples))
        elif self.LossInUse == LOSS_MAX_UNC:
            return _rt_ret(Certainty_Loss()(model, X, y, y_pred, num_samples))
        elif self.LossInUse == LOSS_MIN_BINARYCROSSENT:
            return _rt_ret(BinaryCrossEntropy_Loss()(model, X, y, y_pred, num_samples))
        elif self.LossInUse == LOSS_FOCAL:
            loss, stats = focal_loss(y_pred, y,
                                     gamma=getattr(self, "focal_gamma", 2.0),
                                     alpha=getattr(self, "focal_alpha", 1.0))
            try:
                if getattr(self, "use_wandb", False) and wandb.run is not None:
                    wandb.log({f"FOCAL/{k}": v for k, v in stats.items()},
                              step=getattr(self, "_current_epoch", 0))
            except Exception:
                pass
            return _rt_ret(loss)
        elif self.LossInUse == LOSS_CLUE_LITE:
            loss, stats = clue_lite_loss(y_pred, y,
                                         clue_lambda=getattr(self, "clue_lambda", 0.2),
                                         detach_proxy=getattr(self, "clue_detach_proxy", True))
            try:
                if getattr(self, "use_wandb", False) and wandb.run is not None:
                    wandb.log({f"CLUE/{k}": v for k, v in stats.items()},
                              step=getattr(self, "_current_epoch", 0))
            except Exception:
                pass
            return _rt_ret(loss)
        elif self.LossInUse == LOSS_CLUE:
            loss, stats = clue_loss(model, X, y,
                                    alpha=getattr(self, "clue_alpha", 0.5),
                                    mc_passes=getattr(self, "clue_mc_passes", 5))
            try:
                if getattr(self, "use_wandb", False) and wandb.run is not None:
                    wandb.log({f"CLUE_MC/{k}": v for k, v in stats.items()},
                              step=getattr(self, "_current_epoch", 0))
            except Exception:
                pass
            return _rt_ret(loss)

        # if arrives here, it means that the loss function is the cross entropy loss
        return _rt_ret(CrossEntropy_Loss()(model, X, y, y_pred, num_samples))

    # ------------------------------------------------------------------
    #  Robust training epoch (PGD-AT / TRADES / MART)
    # ------------------------------------------------------------------

    _DATASET_NORM = {
        "cifar10":       ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        "binaryCifar10": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        "cifar100":      ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "svhn":          ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    }
    _IMAGENET_NORM_SMALL = ((0.4810, 0.4574, 0.4078), (0.2146, 0.2104, 0.2138))
    _IMAGENET_NORM_224   = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    class _NormWrap(nn.Module):
        """Temporary wrapper: normalizes [0,1] pixel input before forward."""
        def __init__(self, model, mean_t, std_t):
            super().__init__()
            self.model = model
            self.mean_t = mean_t
            self.std_t = std_t
        def forward(self, x):
            return self.model((x - self.mean_t) / self.std_t)

    def epoch_robust(self, loader, model, opt, method="pgd_at"):
        """One epoch of robust adversarial training."""
        device = self.device
        eps = getattr(self, "robust_eps", 8.0 / 255.0)
        alpha = getattr(self, "robust_alpha", eps / 4.0)
        steps = getattr(self, "robust_steps", 10)
        beta = getattr(self, "robust_beta", 6.0)
        rs = getattr(self, "robust_random_start", True)

        if self.dataset_name == "imageNet":
            _img_orig = os.environ.get("IMAGENET_ORIGINAL", "0").lower() in ("1", "true")
            mean, std = self._IMAGENET_NORM_224 if _img_orig else self._IMAGENET_NORM_SMALL
        else:
            mean, std = self._DATASET_NORM.get(self.dataset_name, ((0,0,0),(1,1,1)))
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        norm_wrap = self._NormWrap(model, mean_t, std_t)

        model.train()
        total_loss, total_err, total_adv_err, n = 0.0, 0.0, 0.0, 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]
            self._rt_add_epoch_imgs(bs)
            _step_tok = self._rt_step_timer_start(bs)

            X_pixel = X * std_t + mean_t
            opt.zero_grad()

            if method == "pgd_at":
                model.eval()
                x_adv_pix = pgd_attack_ce(norm_wrap, X_pixel, y, eps, alpha, steps, rs)
                model.train()
                x_adv_norm = (x_adv_pix - mean_t) / std_t
                logits_adv = model(x_adv_norm)
                loss = F.cross_entropy(logits_adv, y)
            elif method == "trades":
                loss, _ = trades_loss(norm_wrap, X_pixel, y, eps, alpha, steps, beta, rs)
            elif method == "mart":
                loss, _ = mart_loss(norm_wrap, X_pixel, y, eps, alpha, steps, beta, rs)
            else:
                raise ValueError(f"Unknown robust method: {method}")

            loss.backward()
            opt.step()
            self._rt_step_timer_end(_step_tok)

            with torch.no_grad():
                nat_pred = model(X).argmax(1)
                total_err += (nat_pred != y).sum().item()
            total_loss += loss.item() * bs
            n += bs

        return total_err / max(n, 1), total_loss / max(n, 1), []

    def epoch(self, loader, model, opt=None, num_samples=5, lagrangian=None):
        """Standard training/evaluation epoch over the dataset"""
        train_mode = model.training
        if opt is None:
            model.eval()
            torch.set_grad_enabled(False)
        else:
            model.train()
            torch.set_grad_enabled(True)
        total_loss, total_err = 0.,0.
        misclassified_ids = []

        LOG_EVERY = int(os.environ.get("LOG_EVERY", "100"))  # print first few steps and every N iters
        EMPTY_CACHE_EVERY = int(os.environ.get("EMPTY_CACHE_EVERY", "0"))  # 0 disables empty_cache in loop
        _t_batch_start = time.time()
        for ct, (X,y) in enumerate(loader):
            if (not getattr(self, "_debug_shape_printed", False)) and (os.environ.get("DEBUG_IMAGENET_SHAPE", "0").lower() in ("1","true","yes","y")):
                try:
                    print(f"[DEBUG] first batch X.shape={tuple(X.shape)} y.shape={tuple(y.shape)} X.dtype={X.dtype}")
                except Exception:
                    print("[DEBUG] first batch shape print failed")
                self._debug_shape_printed = True
            _fetch_s = time.time() - _t_batch_start
            _t_step0 = time.time()
            X,y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True) # len of bacth size
            if opt:
                self._rt_add_epoch_imgs(X.shape[0])
                _step_tok = self._rt_step_timer_start(X.shape[0])
            else:
                _step_tok = None
            if self.half_prec: 
                # Runs the forward pass with autocasting.
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred = model(X)
                    #loss = nn.CrossEntropyLoss().cuda()(yp,y)

                if opt:   # backpropagation
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        loss = self.LossFunction(model, X, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()

                    if lagrangian is not None:
                        penalty, constraint = lagrangian.get(y_pred)
                        self.scaler.scale(loss+penalty).backward() # Scales the loss, and calls backward() . to create scaled gradients
                    else:
                        self.scaler.scale(loss).backward() # Scales the loss, and calls backward() . to create scaled 
                        
                    self.scaler.step(opt) # Unscales gradients and calls or skips optimizer.step()
                    self.scaler.update()  # Updates the scale for next iteration
                    self._rt_step_timer_end(_step_tok)

            else:
                y_pred = model(X)

                if opt:  # backpropagation
                    opt.zero_grad()

                    loss = self.LossFunction(model, X, y, y_pred, num_samples=num_samples)
                    if lagrangian is not None:
                        penalty, constraint = lagrangian.get(y_pred)
                        (loss + penalty).backward()
                    else:
                        loss.backward()
    
                    opt.step()
                    self._rt_step_timer_end(_step_tok)
            
            misclassified = y_pred.max(dim=1)[1] != y
            # Calculate the IDs of misclassified inputs
            batch_misclassified_ids = (ct * loader.batch_size) + torch.nonzero(misclassified).view(-1)
            misclassified_ids.extend(batch_misclassified_ids.tolist())

            total_err += misclassified.sum().item()
            if opt:  # backpropagation
                total_loss += loss.item() * X.shape[0]
            # --- lightweight progress logging (helps diagnose IO stalls) ---
            if opt and (ct < 5 or ((ct + 1) % max(LOG_EVERY, 1) == 0)):
                try:
                    if getattr(self.device, "type", "") == "cuda":
                        torch.cuda.synchronize()
                    _step_s = time.time() - _t_step0
                    _loss_val = float(loss.detach().item()) if "loss" in locals() else float("nan")
                    _err_frac = float(misclassified.float().mean().item()) if misclassified.numel() > 0 else 0.0
                    _mem_g = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                    _maxmem_g = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                    _imgs_s = float(X.size(0)) / max(_step_s, 1e-6)
                    print(f"[STEP] it={ct:05d} fetch={_fetch_s:.3f}s step={_step_s:.3f}s imgs/s={_imgs_s:.1f} loss={_loss_val:.4f} err={_err_frac:.3f} mem={_mem_g:.2f}G max={_maxmem_g:.2f}G", flush=True)
                except Exception as _e:
                    print(f"[STEP] log failed: {_e}", flush=True)

            del X, y, misclassified, batch_misclassified_ids
            if EMPTY_CACHE_EVERY > 0 and ((ct + 1) % EMPTY_CACHE_EVERY == 0):
                torch.cuda.empty_cache()
            _t_batch_start = time.time()

        torch.set_grad_enabled(True)
        model.train(train_mode)
        return total_err / len(loader.dataset), total_loss / len(loader.dataset), misclassified_ids


    def epoch_interleave_batches(self, loader_wrong, loader_correct, model, opt=None, num_samples=5, weight_loss=None):
        """Standard training/evaluation epoch over the dataset"""
        #total_loss, total_err = 0.,0.
        #misclassified_ids = []
        # ct mod 2 = 0 -> wrong batch
        # ct mod 2 = 1 -> correct batch
        dataloader_iterator_wrong = iter(loader_wrong)
        dataloader_iterator_correct = iter(loader_correct)
        
        no_batches = len(loader_wrong) #+len(loader_correct) # number of batches
        #datasetSize =  len(loader_wrong.dataset) +  len(loader_correct.dataset) # number of inputs in the dataset
        #batch_size = loader_wrong.batch_size

        prints = False
        if dynamic_weights:
            # Define the loss weights as nn.Parameter objects
            weight_loss1 = weight_loss[1] #torch.tensor(0.5, requires_grad=True)
            weight_loss2 = weight_loss[2] #torch.tensor(0.5, requires_grad=True)
            weight_optimizer = weight_loss[0] #optim.Adam([{'params':[weight_loss1, weight_loss2]}],lr=0.01)

        elif Weighted_Sum:
            weight_loss1 = torch.tensor(0.1)
            weight_loss2 = torch.tensor(0.9)

        elif Adaptive_Balancing:
            weight_loss1 = 1-weight_loss
            weight_loss2 = weight_loss



        for ct in range(no_batches):

            X_wrong,y_wrong = next(dataloader_iterator_wrong)
            X_correct,y_correct = next(dataloader_iterator_correct)

            X_wrong,y_wrong = X_wrong.to(self.device), y_wrong.to(self.device) # len of bacth size
            X_correct,y_correct = X_correct.to(self.device), y_correct.to(self.device) # len of bacth size
            _bs = X_wrong.shape[0] + X_correct.shape[0]
            if opt:
                self._rt_add_epoch_imgs(_bs)
                _step_tok = self._rt_step_timer_start(_bs)
            else:
                _step_tok = None

            if self.half_prec: 
                # Runs the forward pass with autocasting.
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred_wrong = model(X_wrong)
                    y_pred_correct = model(X_correct)
                    #loss = nn.CrossEntropyLoss().cuda()(yp,y)

                if opt:   # backpropagation
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        self.LossInUse = LOSS_2nd_stage_wrong
                        loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)

                        self.LossInUse = LOSS_2nd_stage_correct
                        loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)

                        # Calculate the total loss with dynamic weights
                        if Weighted_Sum or Adaptive_Balancing or dynamic_weights:
                            loss = weight_loss1 * loss_wrong + weight_loss2 * loss_correct
                        else:
                            loss = loss_wrong + loss_correct

                    opt.zero_grad()
                    if dynamic_weights: weight_optimizer.zero_grad() 

                    self.scaler.scale(loss).backward() # Scales the loss, and calls backward() . to create scaled gradients
                    self.scaler.step(opt) # Unscales gradients and calls or skips optimizer.step()
                    self.scaler.update()  # Updates the scale for next iteration
                    self._rt_step_timer_end(_step_tok)

                    if dynamic_weights: weight_optimizer.step()


            else:
                #loss = nn.CrossEntropyLoss()(yp,y)
                #torch.autograd.set_detect_anomaly(True)
                y_pred_wrong = model(X_wrong)
                y_pred_correct = model(X_correct)

                if opt:  # backpropagation
                    self.LossInUse = LOSS_2nd_stage_wrong
                    loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)
                    if prints:
                        print(self.LossInUse)
                        print(loss_wrong)

                    self.LossInUse = LOSS_2nd_stage_correct 
                    loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)
                    if prints:
                        print(self.LossInUse)
                        print(loss_correct)

                    # Calculate the total loss with dynamic weights
                    if Weighted_Sum or Adaptive_Balancing or dynamic_weights:
                        loss = weight_loss1 * loss_wrong + weight_loss2 * loss_correct
                    else:
                        loss = loss_wrong + loss_correct

                    # self.LossInUse = LOSS_MIN_CROSSENT
                    # ce_loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)
                    # self.LossInUse = LOSS_MAX_UNC
                    # pe_loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)



                    # self.LossInUse = LOSS_MIN_CROSSENT 
                    # ce_loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)
                    # self.LossInUse = LOSS_MIN_UNC 
                    # pe_loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)

                    # loss = ce_loss_wrong + pe_loss_wrong + ce_loss_correct + pe_loss_correct
                    if prints:
                        print(loss)
                        print()

                    opt.zero_grad()
                    if dynamic_weights: weight_optimizer.zero_grad() 

                    loss.backward()
                    opt.step()
                    self._rt_step_timer_end(_step_tok)
                    if dynamic_weights: weight_optimizer.step()
                        
        return #total_err /datasetSize, total_loss/datasetSize, misclassified_ids


    def epoch_adversarial(self, loader, model, attack, dataset, epsilon=0.1, num_iter=20, alpha=0.01, ratio=1, opt=None, num_samples=5, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.

        #ratio - some have adversarial example and other no
        no_adv = int(ratio * len(loader.dataset))
        no_clean = len(loader.dataset) - no_adv

        l_adv = [True] * no_adv 
        l_clean = [False] * no_clean 
        decision = l_adv + l_clean
        random.shuffle(decision)
        
        grad = 0 #for fgsm with gradient alignment

        # for fgsm free
        delta_real = None
        minibatch_replay = 4 
        counter_minibatch_replay = 0
        X_prev, y_prev =  None, None

        # for fgsm grad alignment
        #we use λ = 0.1 for the CIFAR-10 and λ = 0.5
        if dataset == "cifar10" or dataset == "cifar10-c" or dataset == "binaryCifar10" or dataset == "cifar100" :
            grad_align_cos_lambda = 0.1 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "mnist":
            grad_align_cos_lambda = 0.5 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "imageNet":
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer
        else: #svhn
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer

        ct = 0
        misclassified_ids = []
        for X,y in loader:
            if attack == "fgsm_free" and X_prev is not None and counter_minibatch_replay % minibatch_replay != 0:  
                # take new inputs only each `minibatch_replay` iterations
                X, y = X_prev, y_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm

            X,y = X.to(self.device), y.to(self.device)

            if decision[ct]:
                #adversarial example
                if attack == "fgsm": #adversarial examples fgsm
                    delta = self.fgsm(model, X, y, epsilon=epsilon, num_samples=num_samples, **kwargs) 
                elif attack == "fgsm_rs": #adversarial examples fgsm with random initialization of deltas
                    delta = self.fgsm_rs(model, X, y, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                elif attack == "fgsm_free": #adversarial examples fgsm with random initialization of deltas
                    delta, delta_real = self.fgsm_free(model, X, y, delta_real, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    counter_minibatch_replay += 1

                    if counter_minibatch_replay % minibatch_replay == 0:
                        counter_minibatch_replay = 0
                        X_prev = X.clone()
                        y_prev = y.clone()

                elif attack == "pgd":  #adversarial examples pgd_linf
                    delta = self.pgd_linf(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                elif attack == "pgd_rs":#adversarial examples pgd_linf with random initialization of deltas
                    delta = self.pgd_linf_rs(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                elif attack == "fgsm_grad_align": #adversarial examples fgsm with gradient alignment 
                    delta, grad = self.fgsm_grad_align(model, X, y, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                else: 
                    print("wrong attack")
                    return -1

                X_input = X + delta 
            else:
                #clean data example
                X_input = X 


            if self.half_prec: 
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred = model(X_input)
            else:
                y_pred = model(X_input)

            log_interval = 100 #log every 100 batches

            # gradient alignment
            if decision[ct] and attack == "fgsm_grad_align":
                loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

                if ct % log_interval == 0:
                    v = getattr(self.LossFunction, "pe_rms_v", None)
                    if v is not None:
                        wandb.log({
                            "loss": loss.detach().item(),
                            "pe_rms_v": v.detach().item(),
                        })

                # runs only if it's a adversarial exmaple and the attack is fgsm with gradient alignment 
                reg = torch.zeros(1).cuda(self.device)[0]  # for .item() to run correctly

                grad2 = self.get_input_grad(model, X, y, epsilon, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += grad_align_cos_lambda * (1.0 - cos.mean())
                loss += reg 
            


            if opt: # to train - backpropagation
                if self.half_prec: 
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()  
                else:
                    loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()


            misclassified = y_pred.max(dim=1)[1]  != y

            # Calculate the IDs of misclassified inputs
            batch_misclassified_ids = (ct * loader.batch_size) + torch.nonzero(misclassified).view(-1)
            misclassified_ids.extend(batch_misclassified_ids.tolist())

            total_err += misclassified.sum().item() #error
            if opt: # to train - backpropagation
                total_loss += loss.item() * X.shape[0]  #loss
            ct += 1

            del X, y, misclassified, batch_misclassified_ids
            torch.cuda.empty_cache()

        return total_err / len(loader.dataset), total_loss / len(loader.dataset), misclassified_ids
    

    def epoch_adversarial_interleave_batches(self, loader_wrong, loader_correct, model, attack, dataset, epsilon=0.1, num_iter=20, alpha=0.01, ratio=1, opt=None, num_samples=5, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        #total_loss, total_err = 0.,0.

        dataloader_iterator_wrong = iter(loader_wrong)
        dataloader_iterator_correct = iter(loader_correct)
        
        no_batches = len(loader_wrong) #+len(loader_correct)
        datasetSize =  len(loader_wrong.dataset) +  len(loader_correct.dataset)

        #batch_size = loader_wrong.batch_size

        #ratio - some have adversarial example and other no
        no_adv = int(ratio * datasetSize)
        no_clean = datasetSize - no_adv

        l_adv = [True] * no_adv 
        l_clean = [False] * no_clean 
        decision = l_adv + l_clean
        random.shuffle(decision)
        
        grad = 0 #for fgsm with gradient alignment

        # for fgsm free
        delta_real_wrong = None
        delta_real_correct = None
        minibatch_replay = 4 
        counter_minibatch_replay = 0
        X_wrong_prev, y_wrong_prev =  None, None
        X_correct_prev, y_correct_prev =  None, None

        # for fgsm grad alignment
        #we use λ = 0.1 for the CIFAR-10 and λ = 0.5
        if dataset == "cifar10" or dataset == "cifar10-c" or dataset == "binaryCifar10" or dataset == "cifar100" :
            grad_align_cos_lambda = 0.1 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "mnist":
            grad_align_cos_lambda = 0.5 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "imageNet":
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer
        else: #svhn
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer

        ct = 0
        #misclassified_ids = []
        for ct in range(no_batches):
            X_wrong,y_wrong = next(dataloader_iterator_wrong)
            X_correct,y_correct = next(dataloader_iterator_correct)

            if attack == "fgsm_free" and X_wrong_prev is not None and counter_minibatch_replay % minibatch_replay != 0:  
                # take new inputs only each `minibatch_replay` iterations
                X_correct, y_correct = X_correct_prev, y_correct_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm
                X_wrong, y_wrong = X_wrong_prev, y_wrong_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm

            X_wrong,y_wrong = X_wrong.to(self.device), y_wrong.to(self.device) # len of bacth size
            X_correct,y_correct = X_correct.to(self.device), y_correct.to(self.device) # len of bacth size
        

            if decision[ct]:
                #adversarial example
                if attack == "fgsm": #adversarial examples fgsm
                    delta_wrong = self.fgsm(model, X_wrong, y_wrong, epsilon=epsilon, num_samples=num_samples, **kwargs) 
                    delta_correct = self.fgsm(model, X_correct, y_correct, epsilon=epsilon, num_samples=num_samples, **kwargs) 

                elif attack == "fgsm_rs": #adversarial examples fgsm with random initialization of deltas
                    delta_wrong = self.fgsm_rs(model, X_wrong, y_wrong, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    delta_correct = self.fgsm_rs(model, X_correct, y_correct, epsilon=epsilon, num_samples=num_samples,  **kwargs) 

                elif attack == "fgsm_free": #adversarial examples fgsm with random initialization of deltas
                    delta_wrong, delta_real_wrong = self.fgsm_free(model, X_wrong, y_wrong, delta_real_wrong, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    delta_correct, delta_real_correct = self.fgsm_free(model, X_correct, y_correct, delta_real_correct, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    counter_minibatch_replay += 1

                    if counter_minibatch_replay % minibatch_replay == 0:
                        counter_minibatch_replay = 0
                        X_correct_prev = X_correct.clone()
                        y_correct_prev = y_correct.clone()
                        X_wrong_prev = X_wrong.clone()
                        y_wrong_prev = y_wrong.clone()

                elif attack == "pgd":  #adversarial examples pgd_linf
                    delta_wrong = self.pgd_linf(model, X_wrong, y_wrong, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                    delta_correct = self.pgd_linf(model, X_correct, y_correct, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                    
                elif attack == "pgd_rs":#adversarial examples pgd_linf with random initialization of deltas
                    delta_wrong = self.pgd_linf_rs(model, X_wrong, y_wrong, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                    delta_correct = self.pgd_linf_rs(model, X_correct, y_correct, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs)

                elif attack == "fgsm_grad_align": #adversarial examples fgsm with gradient alignment 
                    delta_wrong, grad_wrong = self.fgsm_grad_align(model, X_wrong, y_wrong, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    delta_correct, grad_correct = self.fgsm_grad_align(model, X_correct, y_correct, epsilon=epsilon, num_samples=num_samples,  **kwargs) 

                else: 
                    print("wrong attack")
                    return -1

                X_correct_input = X_correct + delta_correct 
                X_wrong_input = X_wrong + delta_wrong 
            else:
                #clean data example
                X_correct_input = X_correct 
                X_wrong_input = X_wrong 


            if self.half_prec: 
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred_wrong = model(X_wrong_input)
                    y_pred_correct = model(X_correct_input)
            else:
                y_pred_wrong = model(X_wrong_input)
                y_pred_correct = model(X_correct_input)

            # gradient alignment 
            if decision[ct] and  attack == "fgsm_grad_align":
                loss_wrong = self.LossFunction(model, X_wrong_input, y_wrong, y_pred_wrong, num_samples=num_samples)
                loss_correct = self.LossFunction(model, X_correct_input, y_correct, y_pred_correct, num_samples=num_samples)
                loss = loss_wrong + loss_correct
                X=X_wrong_input+X_correct_input
                y=y_wrong+y_correct

                # runs only if it's a adversarial exmaple and the attack is fgsm with gradient alignment 
                reg = torch.zeros(1).cuda(self.device)[0]  # for .item() to run correctly

                grad2 = self.get_input_grad(model, X, y, epsilon, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += grad_align_cos_lambda * (1.0 - cos.mean())
                loss += reg 
            

            if opt: # to train - backpropagation
                if self.half_prec: 
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        self.LossInUse = LOSS_2nd_stage_wrong
                        loss_wrong = self.LossFunction(model, X_wrong_input, y_wrong, y_pred_wrong, num_samples=num_samples)

                        self.LossInUse = LOSS_2nd_stage_correct
                        loss_correct = self.LossFunction(model, X_correct_input, y_correct, y_pred_correct, num_samples=num_samples)
                                                
                        loss = loss_wrong + loss_correct

                    opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()  
                else:
                    self.LossInUse = LOSS_2nd_stage_wrong
                    loss_wrong = self.LossFunction(model, X_wrong_input, y_wrong, y_pred_wrong, num_samples=num_samples)

                    self.LossInUse = LOSS_2nd_stage_correct 
                    loss_correct = self.LossFunction(model, X_correct_input, y_correct, y_pred_correct, num_samples=num_samples)
                    
                    loss = loss_wrong + loss_correct

                    opt.zero_grad()
                    loss.backward()
                    opt.step()


    def fgsm(self, model, X, y, epsilon=0.1, num_samples=10, CrossEntropyFunction=False):
        """Construct FGSM adversarial examples. For fair ADV eval use CrossEntropyFunction=True."""
        if self.half_prec:
            delta = torch.zeros_like(X, requires_grad=True).cuda()
            delta.requires_grad = True
            X_input = X + delta
            y_pred = model(X_input)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
            self.scaler.scale(loss).backward()
            del X_input, y_pred
            torch.cuda.empty_cache()
        else:
            delta = torch.zeros_like(X, requires_grad=True)
            X_input = X + delta
            y_pred = model(X_input)
            loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
            loss.backward()
            del X_input, y_pred
            torch.cuda.empty_cache()
        return epsilon * delta.grad.detach().sign()


    def fgsm_rs(self, model, X, y, epsilon=0.1, alpha=0.375, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
        delta.requires_grad = True
        X_input = X + delta
        y_pred = model(X_input)
        
        #loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

        #loss = nn.CrossEntropyLoss()(output, y)# + entropy_loss(output)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        delta = delta.detach()
        
        return delta


    def fgsm_free(self, model, X, y, delta, epsilon=0.1, alpha=0.375, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        if delta is None:
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
    
        delta.requires_grad = True
        X_input = X + delta[:X.size(0)]
        y_pred = model(X_input)

        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

        #loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        #loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = torch.max(torch.min(1-X, delta.data[:X.size(0)]), 0-X)
        delta_return = delta.detach()
        
        return delta_return[:X.size(0)], delta


    def fgsm_grad_align(self, model, X, y, epsilon=0.1, alpha=0.375, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
        delta.requires_grad = True
        X_input = X +  delta
        y_pred = model(X_input)

        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)
        #loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        #loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)

        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
        loss.backward()

        #grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        
        delta = delta.detach()
        grad = grad.detach()

        
        return delta, grad


    def pgd_linf(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, num_samples=10, randomize=False, CrossEntropyFunction=False):
        """ Construct FGSM adversarial examples on the examples X"""
        #print(epsilon)
        if self.half_prec: 
            if randomize:
                delta = torch.rand_like(X, requires_grad=True).cuda()
                delta.data = delta.data * 2 * epsilon - epsilon
            else:
                delta = torch.zeros_like(X, requires_grad=True).cuda()
            delta.requires_grad = True

            for t in range(num_iter):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    X_input = X +  delta
                    y_pred = model(X_input)

                    loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
                    #loss = nn.CrossEntropyLoss().cuda()(output, y) #+ entropy_loss(output)

                self.scaler.scale(loss).backward()  
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()    

                del X_input, y_pred
                torch.cuda.empty_cache()

        else:

            if randomize:
                delta = torch.rand_like(X, requires_grad=True)
                delta.data = delta.data * 2 * epsilon - epsilon
            else:
                delta = torch.zeros_like(X, requires_grad=True)
            delta.requires_grad = True
            
            for t in range(num_iter):
                X_input = X + delta
                y_pred = model(X_input)
                
                loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
                loss.backward()

                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()

                #del X_input, y_pred
                #torch.cuda.empty_cache()

        return delta.detach()


    def pgd_linf_rs(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, num_samples=10, randomize=False, CrossEntropyFunction=False):
        """PGD-Linf with random start; same update rule as pgd_linf (attack-all). Linf ball only. For fair ADV eval use CrossEntropyFunction=True."""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)

        for _ in range(num_iter):
            delta.requires_grad = True
            X_input = X + delta
            y_pred = model(X_input)
            loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
            loss.backward()

            delta.data = (delta.data + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad = None

        return delta.detach()


    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


    def pgd_l2(self, model, X, y, epsilon, alpha, num_iter):
        delta = torch.zeros_like(X, requires_grad=True)

        for t in range(num_iter):
            output = model(X + delta)
            loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)
            #loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.data *= epsilon / self.norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
            
        return delta.detach()


    def get_input_grad(self, model, X, y, eps, delta_init='none', num_samples=10, backprop=False):
        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = torch.zeros_like(X).uniform_(-eps, eps)
            delta.requires_grad = True
            #delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
        elif delta_init == 'random_corner':
            delta = torch.zeros_like(X).uniform_(-eps, eps)
            delta.requires_grad = True
            #delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
            delta = eps * torch.sign(delta)
        else:
            raise ValueError('wrong delta init')

        X_input = X +  delta
        y_pred = model(X_input)
                
        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad


    def get_uniform_delta(self, shape, eps, requires_grad=True):
        delta = torch.zeros(shape) #.cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta


    def l2_norm_batch(self, v):
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms


    def testModel_logs(self, dataset_name, models_name, iteration, alg, ratio ,epsilon, numIt, alpha, ratioADV, trainTime, write_pred_logs=False, num_samples=5, calibration=False):

        self.model.eval() # evaluate the model
        
        list_to_write = []

        #test the accuracy of the model
        t3 = time.time()
        test_err, test_loss, test_entropy, test_MI, test_extra = self.test_epoch(
            self.loader.test_loader, self.model,
            num_samples=num_samples, models_name=models_name,
            write_pred_logs=write_pred_logs, iteration=iteration,
            calibration=calibration, return_details=True
        )
        #test_err, test_loss = self.epoch(self.loader.test_loader, self.model)
        testTime = time.time() - t3

        _ratio = 0.0 if isinstance(ratio, list) else ratio # only used for standard training 
            

        str_write = str(iteration) + "," + alg + "," + str(_ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) +  \
                            ",std,0,0,0," + str(test_err) + "," + str(test_loss) + "," + str(testTime) + "," + str(trainTime) + "," + \
                            str(test_entropy) + "," + str(test_MI) + "\n"
        
        list_to_write.append(str_write)


        #test the adversarial accuracy of the model
        if dataset_name == "cifar10":
            #eps_test_list =  [2, 4, 8, 12, 16]
            eps_test_list =  [4]
        elif dataset_name == "cifar10-c":
            #eps_test_list =  [2, 4, 8, 12, 16]
            eps_test_list =  [4]            
        elif dataset_name == "binaryCifar10":
            eps_test_list =  [4]
        elif dataset_name == "cifar100":
            eps_test_list =  [4]            
        elif dataset_name ==   "mnist": #mnist
            #eps_test_list =  [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            eps_test_list =  [0.3]
        elif dataset_name == "imageNet":
            #eps_test_list =  [2, 4, 8]
            eps_test_list =  [4]
        else: #if dataset_name ==   "svhn":
            #eps_test_list =  [2, 4, 8, 12]
            eps_test_list =  [4]


        num_iterTest_list = [20]
        alpha_test_list = [0.01]
        #adv_err, adv_loss, uncertainty = 0., 0., 0.
        # testing the final model using PGD
        for i, eps_test in enumerate(eps_test_list):
            for num_iterTest in num_iterTest_list:
                for alpha_test in alpha_test_list:
                    if dataset_name == "mnist": #mnist
                        _eps_test = eps_test
                    else: #svhn or cifar10 or imageNet
                        _eps_test = eps_test/255.0

                    t4 = time.time()
                    #adv_err, adv_loss = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", _eps_test, num_iterTest, alpha_test, 1, newLoss=False)
                    adv_err, adv_loss, adv_entropy, adv_MI, adv_extra = self.test_epoch_adversarial(
                        self.loader.test_loader, self.model, epsilon=_eps_test, num_iter=num_iterTest, alpha=alpha_test,
                        num_samples=num_samples, models_name=models_name, write_pred_logs=write_pred_logs,
                        iteration=iteration, calibration=calibration, return_details=True, attack_type="pgd_linf"
                    )
                    #if _eps_test == epsilon:
                    #    adv_err, adv_loss, adv_entropy, adv_MI = _adv_err, _adv_loss, _adv_entropy, _adv_MI

                    advTestTime = time.time() - t4
                    str_write = str(iteration) + "," + alg + "," + str(_ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) + \
                                    ",pgd," + str(_eps_test) + "," + str(num_iterTest) + "," + str(alpha_test) + "," + \
                                    str(adv_err) + "," + str(adv_loss) + "," + str(advTestTime) + "," + str(trainTime) + "," + str(adv_entropy) + "," + str(adv_MI) + "\n"

                    list_to_write.append(str_write)


        #write logs
        if isinstance(ratio, list):
            for _ratio_2_test in ratio:
                if _ratio_2_test < 1.0:
                    filename = "./logs1/logs_" + models_name + "_ratio" + str(_ratio_2_test) + ".txt"
                else:
                    filename = "./logs/logs_" + models_name + ".txt"

                if iteration == 1:
                    f = open(filename, "w")
                    f.write("it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime,Entropy,MI\n")
                else:
                    f = open(filename, "a")
        

                for str_write in list_to_write:
                    f.write(str_write)

        else:

            filename = "./logs/logs_" + models_name + ".txt"
            if iteration == 1:
                f = open(filename, "w")
                f.write("it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime,Entropy,MI\n")
            else:
                f = open(filename, "a")


            for str_write in list_to_write:
                f.write(str_write)

        f.close()
        self.model.train() # go back to train mode

        # step = epoch (1, 2, ...); include "epoch" so Wandb can use it as x-axis
        try:
            if wandb.run is not None:
                wandb.log({
                    "epoch": int(iteration),
                    "STD/Error": test_err,
                    "STD/Entropy": test_entropy,
                    "STD/MI": test_MI,
                    "STD/uA": test_extra["uA"],
                    "STD/uAUC": test_extra["uAUC"],
                    "STD/Corr": test_extra["Corr"],
                    "STD/Wasserstein": test_extra["Wasserstein"],
                    "STD/ECE": test_extra["ECE"],
                    "STD/u_thr": test_extra["u_thr"],
                    "STD/AUROC_err_conf": test_extra.get("AUROC_err_conf", float("nan")),
                    "STD/AUROC_err_unc":  test_extra.get("AUROC_err_unc", float("nan")),
                    "PGD/Error": adv_err,
                    "PGD/Entropy": adv_entropy,
                    "PGD/MI": adv_MI,
                    "PGD/uA": adv_extra["uA"],
                    "PGD/uAUC": adv_extra["uAUC"],
                    "PGD/Corr": adv_extra["Corr"],
                    "PGD/Wasserstein": adv_extra["Wasserstein"],
                    "PGD/ECE": adv_extra["ECE"],
                    "PGD/AUROC_err_conf": adv_extra.get("AUROC_err_conf", float("nan")),
                    "PGD/AUROC_err_unc":  adv_extra.get("AUROC_err_unc", float("nan")),
                }, step=int(iteration))
        except Exception:
            pass

        return test_err, test_loss, test_entropy, test_MI, adv_err, adv_loss, adv_entropy, adv_MI

    def _run_extra_evals(self, global_epoch, dataset_name, model, loader):
        """Run ADV suite, C-suite, and LT metrics every eval_extra_every epochs; log to wandb with ADV/, C/, LT/ prefixes."""
        eval_extra_every = int(getattr(self, "eval_extra_every", 0))
        if eval_extra_every <= 0 or (global_epoch % eval_extra_every != 0):
            return

        was_training = model.training
        model.eval()

        eval_adv_suite = getattr(self, "eval_adv_suite", False)
        adv_attacks_str = getattr(self, "adv_attacks", "fgsm,pgd_linf,pgd_linf_rs")
        adv_eps = float(getattr(self, "adv_eps", 8))
        adv_steps = int(getattr(self, "adv_steps", 20))
        adv_pixel = getattr(self, "adv_pixel", True)
        eval_c_suite = getattr(self, "eval_c_suite", False)
        c_corruptions_str = getattr(self, "c_corruptions", "gaussian_noise,brightness")
        c_severities = int(getattr(self, "c_severities", 5))
        imbalance = (getattr(self, "imbalance", "none") or "none").strip().lower()

        to_log = {}
        num_samples = 5

        # ---- ADV suite (all datasets) ----
        if eval_adv_suite and loader.test_loader is not None:
            eps_01 = (adv_eps / 255.0) if adv_pixel else adv_eps
            if dataset_name == "mnist":
                eps_01 = adv_eps  # MNIST often in 0-1
            alpha = (eps_01 * 2.5 / max(adv_steps, 1)) if adv_steps else 0.01
            for attack_name in [a.strip() for a in adv_attacks_str.split(",") if a.strip()]:
                attack_type = "pgd_linf" if attack_name == "pgd_linf" else ("pgd_linf_rs" if attack_name == "pgd_linf_rs" else "fgsm")
                try:
                    adv_err, adv_loss, _, _, _ = self.test_epoch_adversarial(
                        loader.test_loader, model, epsilon=eps_01, num_iter=adv_steps, alpha=alpha,
                        num_samples=num_samples, return_details=True, attack_type=attack_type
                    )
                    prefix = f"ADV/{attack_name}/eps{int(adv_eps)}"
                    if "pgd" in attack_name:
                        prefix += f"/steps{adv_steps}"
                    to_log[f"{prefix}/Error"] = float(adv_err)
                    to_log[f"{prefix}/Acc"] = float(1.0 - adv_err)
                    to_log[f"{prefix}/Loss"] = float(adv_loss)
                except Exception as e:
                    print(f"[extra_evals] ADV {attack_name} failed: {e}", flush=True)

        # ---- C-suite (CIFAR10 / CIFAR100 only) ----
        if eval_c_suite and dataset_name in ("cifar10", "cifar100"):
            data_root = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
            if not os.path.isdir(data_root):
                data_root = "../data"
            sev = min(5, max(1, c_severities))
            test_transform = loader.test_loader.dataset.transform if hasattr(loader.test_loader.dataset, "transform") else None
            if test_transform is None:
                cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
                cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                mean, std = (cifar10_mean, cifar10_std) if dataset_name == "cifar10" else (cifar100_mean, cifar100_std)
                test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            for corr in [c.strip() for c in c_corruptions_str.split(",") if c.strip()]:
                try:
                    if dataset_name == "cifar10":
                        C_dataset = CIFAR10C(data_root, name=corr, transform=test_transform)
                    else:
                        C_dataset = CIFAR100C(data_root, name=corr, transform=test_transform)
                    n_total = len(C_dataset)
                    n_per_sev = 10000
                    if n_total >= n_per_sev * 5:
                        indices = list(range((sev - 1) * n_per_sev, min(sev * n_per_sev, n_total)))
                        C_dataset = Subset(C_dataset, indices)
                    c_loader = DataLoader(C_dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0)
                    err, loss, _, _, _ = self.test_epoch(c_loader, model, num_samples=num_samples, return_details=True)
                    to_log[f"C/{corr}/s{sev}/Error"] = float(err)
                    to_log[f"C/{corr}/s{sev}/Acc"] = float(1.0 - err)
                    to_log[f"C/{corr}/s{sev}/Loss"] = float(loss)
                except Exception as e:
                    print(f"[extra_evals] C {corr} failed: {e}", flush=True)

        # ---- LT metrics (RunB only): log only when imbalance != none (RunA must not produce LT/) ----
        if imbalance != "none" and dataset_name in ("cifar10", "cifar100", "svhn"):
            try:
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for X, y in loader.test_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        out = model(X)
                        pred = out.argmax(dim=1)
                        all_preds.append(pred.cpu().numpy())
                        all_labels.append(y.cpu().numpy())
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                n_classes = int(getattr(loader, "num_classes", all_labels.max() + 1))
                per_class_correct = np.zeros(n_classes)
                per_class_total = np.zeros(n_classes)
                for c in range(n_classes):
                    mask = all_labels == c
                    per_class_total[c] = mask.sum()
                    if per_class_total[c] > 0:
                        per_class_correct[c] = (all_preds[mask] == all_labels[mask]).sum()
                per_class_acc = np.where(per_class_total > 0, per_class_correct / per_class_total, 0.0)
                balanced_acc = float(np.mean(per_class_acc[per_class_total > 0])) if (per_class_total > 0).any() else 0.0
                third = max(1, n_classes // 3)
                many_acc = float(np.mean(per_class_acc[:third])) if third else 0.0
                medium_acc = float(np.mean(per_class_acc[third:2*third])) if 2*third > third else 0.0
                few_acc = float(np.mean(per_class_acc[2*third:])) if n_classes > 2*third else 0.0
                to_log["LT/BalancedAcc"] = balanced_acc
                to_log["LT/ManyAcc"] = many_acc
                to_log["LT/MediumAcc"] = medium_acc
                to_log["LT/FewAcc"] = few_acc
            except Exception as e:
                print(f"[extra_evals] LT failed: {e}", flush=True)

        if to_log:
            to_log["epoch"] = int(global_epoch)
            try:
                if wandb.run is not None:
                    wandb.log(to_log, step=int(global_epoch))
                    print(f"[W&B] extra evals logged at epoch {global_epoch}: {list(to_log.keys())}", flush=True)
            except Exception as e:
                print("[W&B extra_evals log skipped]", e, flush=True)

        if was_training:
            model.train()

    def MCdropout(self, model, X, y, num_samples=10, calibration=False):
    #def MCdropout(self, model, X, y, num_samples=10, adversarial=False, epsilon=0.1, num_iter=20, alpha=0.01, **kwargs):
        #MC DROPOUT - UNCERTAINTY
        probs = None
        mean_entropy = None
        num_clases = None

        for n in range(num_samples):
            #need to enable dropout
            model.eval() # evaluate the model

            if self.deep_ensemble: # ensemble as already the softmax applied
                softmax_output = model(X)
                
            else:
                if hasattr(model, "enable_dropout"):
                    model.enable_dropout()
                softmax_output = F.softmax(model(X), dim=1)
            
                if calibration and self.isCalibrated:
                    softmax_output = self.predict_proba(softmax_output)


            if num_clases is None: num_clases = len(softmax_output[0])
            if probs is None: probs = torch.zeros_like(softmax_output)
            probs = probs + softmax_output
            #probs = (probs+F.softmax(output, dim=1)) if probs is not None else F.softmax(output, dim=1) 

            _entropy = entropy(softmax_output)
            if mean_entropy is None: mean_entropy = torch.zeros_like(_entropy)
            mean_entropy = mean_entropy + _entropy
            #mean_entropy = (mean_entropy+entropy(output)) if mean_entropy is not None else entropy(output)

            del softmax_output
            torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())
            #print("\n")


        probs /= float(num_samples) 
        mask = probs == 0  # Create a mask of zero values
        probs[mask] = 10e-20  # Replace zero values with small_value

        log_probs = torch.log(probs)
        entropy_vals = -torch.sum(probs * log_probs, dim=-1)# predictive entropy PE=H

        mean_entropy /= float(num_samples)
        mutual_information_vals = entropy_vals - mean_entropy
        #print(mean_entropy)
        #print(mutual_information_vals)

        #max entropy - uniform distrbution
        #https://math.stackexchange.com/questions/1156404/entropy-of-a-uniform-distribution
        entropy_max = np.log(num_clases)
        normalized_entropy = entropy_vals / entropy_max # normalized predictive entropy
        normalized_mutual_information = mutual_information_vals / entropy_max # normalized mutual information
        #mutual_information is maximum when the second term is 0 and the first is maxinum entropy (uniform distirbution)

        return normalized_entropy, normalized_mutual_information

    """Comparison of Adam-EUAT against the baselines using different evaluation metrics"""
    @staticmethod
    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.sqrt((a*a).mean()) * np.sqrt((b*b).mean()) + 1e-12)
        return float((a*b).mean() / denom)

    @staticmethod
    def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
        # 1D Wasserstein distance via quantile matching
        if len(x) == 0 or len(y) == 0:
            return float("nan")
        x = np.sort(x.astype(np.float64))
        y = np.sort(y.astype(np.float64))
        n = min(len(x), len(y))
        # downsample to same length
        xi = x[np.linspace(0, len(x)-1, n).astype(int)]
        yi = y[np.linspace(0, len(y)-1, n).astype(int)]
        return float(np.mean(np.abs(xi - yi)))

    @staticmethod
    def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
        # Expected Calibration Error
        conf = conf.astype(np.float64)
        correct = correct.astype(np.float64)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if mask.sum() == 0:
                continue
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece += (mask.mean()) * abs(acc_bin - conf_bin)
        return float(ece)

    @staticmethod
    def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """AUROC for binary labels (y_true in {0,1}). Handles ties via average ranks.

        If only one class is present, returns NaN.
        """
        y_true = np.asarray(y_true).astype(np.int32)
        y_score = np.asarray(y_score).astype(np.float64)
        if y_true.size == 0:
            return float("nan")
        pos = (y_true == 1)
        neg = (y_true == 0)
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        order = np.argsort(y_score, kind="mergesort")
        s_sorted = y_score[order]
        ranks = np.empty_like(s_sorted, dtype=np.float64)

        # Average ranks for ties
        n = len(s_sorted)
        i = 0
        while i < n:
            j = i + 1
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            # ranks are 1..n
            avg_rank = 0.5 * ((i + 1) + j)
            ranks[i:j] = avg_rank
            i = j

        ranks_full = np.empty_like(ranks)
        ranks_full[order] = ranks
        sum_pos = float(ranks_full[pos].sum())
        auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
        return float(auc)

    @staticmethod
    def _u_metrics_from_uncertainty(unc: np.ndarray, correct: np.ndarray):
        """
        Confusion by threshold t:
        certain  : unc <= t
        uncertain: unc >  t
        TC = correct & certain
        FU = correct & uncertain
        FC = wrong   & certain
        TU = wrong   & uncertain

        uA(t)  = (TC + TU) / N   (paper's "uncertainty accuracy" style)
        uAUC   = AUC of TCR vs FCR across thresholds
        TCR = TC / (TC + FU)   (among correct, how many are certain)
        FCR = FC / (FC + TU)   (among wrong, how many are (badly) certain)
        """
        unc = unc.astype(np.float64)
        correct = correct.astype(bool)
        wrong = ~correct
        N = len(unc)

        # sweep thresholds over sorted unique values (fast enough for CIFAR10)
        thr = np.unique(unc)
        # guard: if all same uncertainty
        if len(thr) == 1:
            t = thr[0]
            certain = unc <= t
            TC = np.sum(correct & certain)
            TU = np.sum(wrong & (~certain))
            uA = (TC + TU) / max(N, 1)
            return float(uA), float("nan"), float(t), ([], [])

        uA_list = []
        TCR_list = []
        FCR_list = []

        for t in thr:
            certain = unc <= t
            TC = np.sum(correct & certain)
            FU = np.sum(correct & (~certain))
            FC = np.sum(wrong & certain)
            TU = np.sum(wrong & (~certain))

            uA = (TC + TU) / max(N, 1)
            uA_list.append(uA)

            TCR = TC / max(TC + FU, 1)
            FCR = FC / max(FC + TU, 1)
            TCR_list.append(TCR)
            FCR_list.append(FCR)

        # best uA threshold
        idx_best = int(np.argmax(uA_list))
        best_uA = float(uA_list[idx_best])
        best_t = float(thr[idx_best])

        # uAUC: integrate TCR(FCR) after sorting by FCR
        FCR_arr = np.array(FCR_list, dtype=np.float64)
        TCR_arr = np.array(TCR_list, dtype=np.float64)
        order = np.argsort(FCR_arr)
        FCR_arr = FCR_arr[order]
        TCR_arr = TCR_arr[order]
        uAUC = float(np.trapz(TCR_arr, FCR_arr))  # area under curve

        return best_uA, uAUC, best_t, (FCR_arr.tolist(), TCR_arr.tolist())

    def test_epoch(self, loader, model, num_samples=10, models_name=None,
                write_pred_logs=False, iteration=-1, calibration=False,
                return_details=False):

        total_loss, total_err, total_entropy, total_mutual_information, counter_inputs = 0., 0., 0., 0., 0.
        write_scores = True if (models_name is not None and 'binary' in models_name) else False
        lossfunc = nn.CrossEntropyLoss(reduction='none')

        unc_all, correct_all, conf_all = [], [], []

        with torch.no_grad():
            for X, y in loader:
                _data = []
                X, y = X.to(self.device), y.to(self.device)
                counter_inputs += len(y)

                y_pred = model(X)

                # --- choose probs for metrics/logging ---
                if self.deup:
                    probs_for_metrics = F.softmax(y_pred, dim=1)
                    total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                    normalized_entropy = self.deup_model.predict(X).t()[0]
                    normalized_mutual_information = lossfunc(y_pred, y)

                elif self.deep_ensemble:
                    probs_for_metrics = y_pred  # already probs
                    total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                    normalized_entropy, normalized_mutual_information = self.MCdropout(
                        model, X, y, num_samples=1, calibration=calibration
                    )

                elif calibration and self.isCalibrated:
                    probs = self.predict_proba(F.softmax(y_pred, dim=1))
                    probs_for_metrics = probs
                    total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                    normalized_entropy, normalized_mutual_information = self.MCdropout(
                        model, X, y, num_samples=num_samples, calibration=calibration
                    )

                else:
                    probs_for_metrics = F.softmax(y_pred, dim=1)
                    total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                    normalized_entropy, normalized_mutual_information = self.MCdropout(
                        model, X, y, num_samples=num_samples, calibration=calibration
                    )

                # --- loss / entropy / MI totals ---
                loss_batch = lossfunc(y_pred, y)
                total_loss += loss_batch.sum().item()
                total_entropy += normalized_entropy.sum().item()
                total_mutual_information += normalized_mutual_information.sum().item()

                # --- collect arrays for uncertainty metrics ---
                pred = probs_for_metrics.max(dim=1)[1]
                correct_batch = (pred == y).detach().cpu().numpy().astype(np.bool_)
                unc_batch = normalized_entropy.detach().cpu().numpy().astype(np.float32)
                conf_batch = probs_for_metrics.max(dim=1)[0].detach().cpu().numpy().astype(np.float32)

                unc_all.append(unc_batch)
                correct_all.append(correct_batch)
                conf_all.append(conf_batch)

                # --- existing prediction logs (keep) ---
                if write_pred_logs and models_name is not None:
                    _entropy_normalized = normalized_entropy.tolist()
                    _normalized_mutual_information = normalized_mutual_information.tolist()
                    _predictions = pred.tolist()
                    _y = y.tolist()

                    if write_scores:
                        _probs = probs_for_metrics.tolist()

                    for i in range(len(y)):
                        if write_scores:
                            _probs_str = ""
                            for probs_ in _probs[i]:
                                _probs_str += str(probs_) + ':'
                            _data.append((iteration, _y[i], _predictions[i], _y[i]==_predictions[i],
                                        total_err/counter_inputs, total_loss/counter_inputs,
                                        _entropy_normalized[i], total_entropy/counter_inputs,
                                        _normalized_mutual_information[i], total_mutual_information/counter_inputs,
                                        _probs_str[:-1]))
                        else:
                            _data.append((iteration, _y[i], _predictions[i], _y[i]==_predictions[i],
                                        total_err/counter_inputs, total_loss/counter_inputs,
                                        _entropy_normalized[i], total_entropy/counter_inputs,
                                        _normalized_mutual_information[i], total_mutual_information/counter_inputs))

                    self.write_logs_prediction(_data, ('STD1' if write_pred_logs==2 else 'STD') + models_name)

                del X, y, y_pred
                torch.cuda.empty_cache()

        # --- after ALL batches: compute details ---
        err = total_err / len(loader.dataset)
        loss = total_loss / len(loader.dataset)
        ent = total_entropy / len(loader.dataset)
        mi = total_mutual_information / len(loader.dataset)

        if not return_details:
            return err, loss, ent, mi

        unc = np.concatenate(unc_all, axis=0)
        corr = np.concatenate(correct_all, axis=0)
        conf = np.concatenate(conf_all, axis=0)

        err01 = (~corr).astype(np.float32)
        Corr = self._pearson_corr(unc, err01)
        Wass = self._wasserstein_1d(unc[corr], unc[~corr])
        ECE = self._ece(conf, corr, n_bins=15)
        uA, uAUC, best_thr, self._curve = self._u_metrics_from_uncertainty(unc, corr)

        AUROC_conf = self._auroc(err01, 1.0 - conf)
        AUROC_unc  = self._auroc(err01, unc)

        extra = {"uA": uA, "uAUC": uAUC, "Corr": Corr, "Wasserstein": Wass, "ECE": ECE, "u_thr": best_thr, "AUROC_err_conf": AUROC_conf, "AUROC_err_unc": AUROC_unc}
        return err, loss, ent, mi, extra

    def test_epoch_adversarial(
        self,
        loader,
        model,
        epsilon=0.1,
        num_iter=20,
        alpha=0.01,
        num_samples=10,
        models_name=None,
        write_pred_logs=False,
        iteration=-1,
        calibration=False,
        return_details=False,
        attack_type="pgd_linf",
        **kwargs,
    ):
        """Adversarial evaluation over the dataset. attack_type: fgsm, pgd_linf, pgd_linf_rs."""

        total_loss, total_err, total_entropy, total_mutual_information, counter_inputs = 0.0, 0.0, 0.0, 0.0, 0.0
        write_scores = True if (models_name is not None and "binary" in models_name) else False
        lossfunc = nn.CrossEntropyLoss(reduction="none")

        # collect arrays for uncertainty metrics
        unc_all, correct_all, conf_all = [], [], []

        for X, y in loader:
            _data = []
            X, y = X.to(self.device), y.to(self.device)
            counter_inputs += len(y)

            # ---- adversarial examples: dispatch by attack_type (all use CE for fair ADV eval) ----
            if attack_type == "fgsm":
                delta = self.fgsm(model, X, y, epsilon=epsilon, num_samples=num_samples, CrossEntropyFunction=True, **kwargs)
            elif attack_type == "pgd_linf_rs":
                delta = self.pgd_linf_rs(
                    model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha,
                    num_samples=num_samples, CrossEntropyFunction=True, **kwargs
                )
            else:  # pgd_linf default
                delta = self.pgd_linf(
                    model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha,
                    num_samples=num_samples, CrossEntropyFunction=True, **kwargs
                )
            X_input = X + delta

            # forward on adversarial inputs
            y_pred = model(X_input)

            # ---- choose branch + compute uncertainty ----
            # NOTE: probs_for_metrics is what we use for pred/conf (ECE)
            probs_for_metrics = None

            if self.deup:
                # prediction probs
                probs_for_metrics = F.softmax(y_pred, dim=1)
                if write_scores:
                    probs = probs_for_metrics
                total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                # IMPORTANT: use adversarial inputs for uncertainty in adv eval
                normalized_entropy = self.deup_model.predict(X_input).t()[0]
                normalized_mutual_information = lossfunc(y_pred, y)

            elif self.deep_ensemble:
                # deep ensemble output is already probabilities
                probs_for_metrics = y_pred
                if write_scores:
                    probs = probs_for_metrics
                total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                # use adversarial inputs for uncertainty
                normalized_entropy, normalized_mutual_information = self.MCdropout(
                    model, X_input, y, num_samples=1, calibration=calibration
                )

            elif calibration and self.isCalibrated:
                # calibrated probabilities
                probs = self.predict_proba(F.softmax(y_pred, dim=1))
                probs_for_metrics = probs
                total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                normalized_entropy, normalized_mutual_information = self.MCdropout(
                    model, X_input, y, num_samples=num_samples, calibration=calibration
                )

            else:
                probs_for_metrics = F.softmax(y_pred, dim=1)
                if write_scores:
                    probs = probs_for_metrics
                total_err += (probs_for_metrics.max(dim=1)[1] != y).sum().item()

                normalized_entropy, normalized_mutual_information = self.MCdropout(
                    model, X_input, y, num_samples=num_samples, calibration=calibration
                )

            # ---- loss / totals ----
            loss_batch = lossfunc(y_pred, y)
            total_loss += loss_batch.sum().item()

            total_entropy += normalized_entropy.sum().item()
            total_mutual_information += normalized_mutual_information.sum().item()

            # ---- collect arrays for extra metrics (uA/uAUC/Corr/Wass/ECE) ----
            pred = probs_for_metrics.max(dim=1)[1]
            correct_batch = (pred == y).detach().cpu().numpy().astype(np.bool_)
            unc_batch = normalized_entropy.detach().cpu().numpy().astype(np.float32)
            conf_batch = probs_for_metrics.max(dim=1)[0].detach().cpu().numpy().astype(np.float32)

            unc_all.append(unc_batch)
            correct_all.append(correct_batch)
            conf_all.append(conf_batch)

            # ---- write prediction logs (optional) ----
            if write_pred_logs and models_name is not None:
                _entropy_normalized = normalized_entropy.tolist()
                _normalized_mutual_information = normalized_mutual_information.tolist()
                _predictions = pred.tolist()
                _y = y.tolist()

                if write_scores:
                    _probs = probs_for_metrics.tolist()

                for i in range(len(_y)):
                    if write_scores:
                        _probs_str = ""
                        for probs_ in _probs[i]:
                            _probs_str += str(probs_) + ":"

                        _data.append(
                            (
                                iteration,
                                _y[i],
                                _predictions[i],
                                _y[i] == _predictions[i],
                                total_err / counter_inputs,
                                total_loss / counter_inputs,
                                _entropy_normalized[i],
                                total_entropy / counter_inputs,
                                _normalized_mutual_information[i],
                                total_mutual_information / counter_inputs,
                                _probs_str[:-1],
                            )
                        )
                    else:
                        _data.append(
                            (
                                iteration,
                                _y[i],
                                _predictions[i],
                                _y[i] == _predictions[i],
                                total_err / counter_inputs,
                                total_loss / counter_inputs,
                                _entropy_normalized[i],
                                total_entropy / counter_inputs,
                                _normalized_mutual_information[i],
                                total_mutual_information / counter_inputs,
                            )
                        )

                if write_pred_logs == 2:
                    self.write_logs_prediction(_data, "ADV1" + models_name)
                else:
                    self.write_logs_prediction(_data, "ADV" + models_name)

            # cleanup
            del X, y, delta, X_input, y_pred, loss_batch, normalized_mutual_information, normalized_entropy
            torch.cuda.empty_cache()

        # ---- finalize epoch averages ----
        adv_err = total_err / len(loader.dataset)
        adv_loss = total_loss / len(loader.dataset)
        adv_entropy = total_entropy / len(loader.dataset)
        adv_mi = total_mutual_information / len(loader.dataset)

        if not return_details:
            return adv_err, adv_loss, adv_entropy, adv_mi

        # ---- compute extra metrics after full loader ----
        unc = np.concatenate(unc_all, axis=0)
        corr = np.concatenate(correct_all, axis=0)
        conf = np.concatenate(conf_all, axis=0)

        err01 = (~corr).astype(np.float32)
        Corr = self._pearson_corr(unc, err01)
        Wass = self._wasserstein_1d(unc[corr], unc[~corr])
        ECE = self._ece(conf, corr, n_bins=15)
        uA, uAUC, best_thr, _curve = self._u_metrics_from_uncertainty(unc, corr)

        AUROC_conf = self._auroc(err01, 1.0 - conf)
        AUROC_unc  = self._auroc(err01, unc)

        adv_extra = {
            "uA": uA,
            "uAUC": uAUC,
            "Corr": Corr,
            "Wasserstein": Wass,
            "ECE": ECE,
            "u_thr": best_thr,
            "AUROC_err_conf": AUROC_conf,
            "AUROC_err_unc": AUROC_unc,
        }

        return adv_err, adv_loss, adv_entropy, adv_mi, adv_extra

    def write_logs_prediction(self, data, models_name):

        filename = "./logs/predictions_" + models_name + ".txt"
        f = open(filename, "a")
        for pred_data in data:
            str_write = ''
            for dd in pred_data:
                str_write += str(dd) + ','
            f.write(str_write[:-1] + '\n')

        f.close()


    def calibrate(self, loader_data, model, num_samples=5):
        """calibrate probablities training/evaluation after training"""
        if not PlattScaling_Flag  and not IsotonicRegression_Flag and not TemperatureScaling_Flag and not BetaCalibration_Flag:
            return
        
        #calidation set for calibration
        size = int(len(loader_data)*0.1)
        ids = random.sample(range(int(len(loader_data))), size)
        subset = Subset(loader_data, ids)
        sub_loader = DataLoader(subset, batch_size=32 , shuffle=True) 

        #y_pred_list, y_list = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        y_pred_list, y_list = np.array([]), np.array([])

        with torch.no_grad():
            for X,y in sub_loader:
                scores = None
                X,y = X.to(self.device), y.to(self.device) # len of bacth size
                for n in range(num_samples):
                    #need to enable dropout
                    model.eval() # evaluate the model
                    if hasattr(model, "enable_dropout"):
                        model.enable_dropout()

                    if self.half_prec: 
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            softmax_output = F.softmax(model(X), dim=1)
                    else:
                        softmax_output = F.softmax(model(X), dim=1)
                    
                    if scores is None: scores = torch.zeros_like(softmax_output)
                    scores += softmax_output

                scores /= float(num_samples) 

                y_pred_list = np.concatenate((y_pred_list, scores.cpu().numpy()), axis=0) if len(y_pred_list)>0 else scores.cpu().numpy()
                y_list = np.concatenate((y_list, y.cpu().numpy()), axis=0) if len(y_list)>0 else y.cpu().numpy()


        # method (str, default: "mle") – 
        # ‘mle’: Maximum likelihood estimate without uncertainty using a convex optimizer. 
        # ‘momentum’: MLE estimate using Momentum optimizer for non-convex optimization. 
        # ‘variational’: Variational Inference with uncertainty. 
        # ‘mcmc’: Markov-Chain Monte-Carlo sampling with uncertainty.
        method = 'mle'
        #method = 'momentum'
        #method = 'variational'
        #method = 'mcmc'

        if TemperatureScaling_Flag:
            self.calibration = TemperatureScaling(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        elif BetaCalibration_Flag:
            self.calibration = BetaCalibration(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        elif IsotonicRegression_Flag:
            self.calibration = IsotonicRegression(detection=False)
            self.calibration.fit(y_pred_list, y_list)
        
        else:
            self.calibration = LogisticCalibration(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        self.isCalibrated = True
        return 
        

    def calibrate_adversarial(self, loader_data, model, num_samples=5, attack="fgsm", epsilon=0.1, num_iter=20, alpha=0.01):
        """calibrate probablities training/evaluation after training"""
        if not PlattScaling_Flag  and not IsotonicRegression_Flag and not TemperatureScaling_Flag:
            return
        
        #calidation set for calibration
        size = int(len(loader_data)*0.1)
        ids = random.sample(range(int(len(loader_data))), size)
        subset = Subset(loader_data, ids)
        sub_loader = DataLoader(subset, batch_size=32 , shuffle=True) 

        #y_pred_list, y_list = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        y_pred_list, y_list = np.array([]), np.array([])

        for X,y in sub_loader:
            scores = None
            X,y = X.to(self.device), y.to(self.device) # len of bacth size
            #adversarial example
            if attack == "fgsm": #adversarial examples fgsm
                delta = self.fgsm(model, X, y, epsilon=epsilon, num_samples=num_samples) 
            else:
                delta = self.pgd_linf(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples) 
            X_input = X + delta 

            for n in range(num_samples):
                #need to enable dropout
                model.eval() # evaluate the model
                if hasattr(model, "enable_dropout"):
                    model.enable_dropout()

                if self.half_prec: 
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        softmax_output = F.softmax(model(X_input), dim=1)
                else:
                    softmax_output = F.softmax(model(X_input), dim=1)
                
                if scores is None: scores = torch.zeros_like(softmax_output)
                scores += softmax_output.detach()

            scores /= float(num_samples) 

            y_pred_list = np.concatenate((y_pred_list, scores.cpu().numpy()), axis=0) if len(y_pred_list)>0 else scores.cpu().numpy()
            y_list = np.concatenate((y_list, y.cpu().numpy()), axis=0) if len(y_list)>0 else y.cpu().numpy()


        # method (str, default: "mle") – 
        # ‘mle’: Maximum likelihood estimate without uncertainty using a convex optimizer. 
        # ‘momentum’: MLE estimate using Momentum optimizer for non-convex optimization. 
        # ‘variational’: Variational Inference with uncertainty. 
        # ‘mcmc’: Markov-Chain Monte-Carlo sampling with uncertainty.
        method = 'mle'
        #method = 'momentum'
        #method = 'variational'
        #method = 'mcmc'

        if TemperatureScaling_Flag:
            self.calibration = TemperatureScaling(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        elif BetaCalibration_Flag:
            self.calibration = BetaCalibration(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        elif IsotonicRegression_Flag:
            self.calibration = IsotonicRegression(detection=False)
            self.calibration.fit(y_pred_list, y_list)
        
        else:
            self.calibration = LogisticCalibration(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        self.isCalibrated = True
        return 


    def predict_proba(self, scores):

        if not self.ToCalibrate and not self.isCalibrated:
            softmax_output = F.softmax(scores, dim=1)

        else:
            if PlattScaling_Flag or IsotonicRegression_Flag or TemperatureScaling_Flag or BetaCalibration_Flag:
                with torch.no_grad():
                    out = self.calibration.transform(scores.cpu().numpy())
                    softmax_output = torch.tensor(out, device=self.device)
                    
                    if len(softmax_output.shape)==1:
                        # binary case - compute the other probablity 
                        softmax_output = torch.cat((torch.zeros_like(softmax_output.unsqueeze(-1)), softmax_output.unsqueeze(-1)), dim=1)
                        softmax_output[:,0] = 1.0-softmax_output[:,1]
            else:
                softmax_output = F.softmax(scores, dim=1)
        return softmax_output


class model(trainModel):
    def __init__(self, dataset, dataset_name, device, devices_id, lr=0.1, momentum=0, lr_adv=0.1, momentum_adv=0, batch_adv=100, half_prec=False, variants=None):
        self.ecg_lam = 1.0
        self.ecg_tau = 0.7
        self.ecg_k = 10.0
        self.ecg_conf_type = "pmax"
        
        self.loader = dataset
        self.dataset_name = dataset_name 
        self.lr = lr 
        self.momentum = momentum 

        self.lr_adv = lr_adv 
        self.momentum_adv = momentum_adv 
        self.batch_adv = batch_adv

        self.devices_id = devices_id

        super().__init__(device, half_prec=half_prec, variants=variants) #initialize the datasets
    
        self.model = self.resetModel() #initialize model

        dampening=0
        weight_decay=0 if self.dataset_name != "imageNet" else 0.0001
        nesterov=False

        self.use_wandb = (wandb.run is not None)  # or True if you always want it on

        self.opt = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        # hyper-parameters clean data: learning rate, momentum, dampening, weight_decay, nesterov
        # hyper-parameters FGSM: epsilon, ratio
        # hyper-parameters PGD: epsilon, num_iter, alpha, ratio

        #if self.dataset_name == "svhn" or self.dataset_name == "imageNet":
        #    # we only parallelize these 2 datasets
        #    self.model = self.parallelizeModel(self.model)


    def run(self, modelName, iterations=10, stop="epochs", ckptName=None, runName=None):
        self.model.train()
        if ckptName is None: ckptName = modelName
        if runName  is None: runName  = modelName
        return self._train_epochs(modelName, iterations=iterations, stop=stop, ckptName=ckptName, runName=runName)

    def _train_epochs(self, modelName, iterations=10, stop="epochs", ckptName=None, runName=None):
        ''' uploads the models or trains it from scratch'''
        if ckptName is None: ckptName = modelName
        if runName  is None: runName  = modelName

        path = "./models/" + modelName
        if "binaryCifar10" in modelName:
            _dataset = "binaryCifar10"
        elif "cifar100" in modelName:
            _dataset = "cifar100"
        elif "cifar10-c" in modelName:
            _dataset = "cifar10-c"   
        elif "cifar10" in modelName:
            _dataset = "cifar10"
        elif "mnist" in modelName:
            _dataset = "mnist"
        elif "imageNet" in modelName:
            _dataset = "imageNet"
        else: #svhn
            _dataset = "svhn"

        aux = modelName.split("_")

        if  "robust" in aux[1] and aux[2] == "FGSM":
            _ratio =  float(aux[4][5:])
            ratio_adv = float(aux[5][8:])

            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 or cifar100 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            print('train with clean data and then with fgsm (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])))

            trainTime, train_err, train_loss = self.standard_fgsm_train(self.model, modelName, self.loader, _dataset, self.opt, ratio=_ratio, 
                        eps_train=_eps_train, iterations=iterations, ratio_adv=ratio_adv, lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
                

        elif "robust" in aux[1] and aux[2] == "PGD":
            _ratio = float(aux[6][5:])
            ratio_adv = float(aux[7][8:])

            _num_iterTrain = int(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 ro cifar100 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            _alpha_train =float( aux[5][5:])

            print('train with clean data and then with pgd (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ', no_ite=' + str(_num_iterTrain) + ', alpha=' + str(_alpha_train))
            
            trainTime, train_err, train_loss = self.standard_pgd_train(self.model, modelName, self.loader, _dataset, self.opt, ratio=_ratio, 
                        num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, iterations=iterations, ratio_adv=ratio_adv,
                        lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)


        else:
            #train all model with checkpointing and early stopping (only used fo standard training to differentiate the full standard training and pre-training) 
                trainTime, train_err, train_loss = self.standard_train(
                        self.model,
                        runName,
                        self.loader,
                        _dataset,
                        self.opt,
                        iterations=iterations,
                        ckptName=ckptName,
                        runName=runName
                    )
        return trainTime, train_err, train_loss

    def testModel(self):
        '''test the model at the end to evaluate standard accuracy'''
        # switch to evaluate mode
        self.model.eval()

        t3 = time.time()
        test_err, test_loss,_ = self.epoch(self.loader.test_loader, self.model)
        testTime = time.time() - t3

        return (testTime, test_err, test_loss)


    def testModel_adversarial_pgd_all(self):
        '''test the model with adversarial examples at the end to evaluate adversarial accuracy
            using different sets of hyperparameter'''
        # switch to evaluate mode
        self.model.eval()

        eps_test_list =  [0.01, 0.05, 0.1, 0.2]
        num_iterTest_list = [10, 20]
        alpha_test_list = [0.01]

        advTestTime_pgd = []
        adv_loss_pgd = []
        adv_err_pgd = []

        # testing the final model using PGS
        for eps_test in eps_test_list:
            for num_iterTest in num_iterTest_list:
                for alpha_test in alpha_test_list:
                    print("PGD test epsilon test= " + str(eps_test) + "  num_iterTest = " + str(num_iterTest)  + "  alpha_test = " + str(alpha_test))

                    t4 = time.time()
                    adv_err, adv_loss,_ = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", eps_test, num_iterTest, alpha_test, 1)
                    advTestTime = time.time() - t4

                    advTestTime_pgd.append(advTestTime)
                    adv_loss_pgd.append(adv_loss)
                    adv_err_pgd.append(adv_err)

        return (advTestTime_pgd, adv_err_pgd, adv_loss_pgd)


    def testModel_adversarial_pgd(self, eps_test, num_iterTest, alpha_test):
        '''test the model with adversarial examples at the end to evaluate adversarial accuracy
            using fixing hyperparameter''' 
        # switch to evaluate mode
        self.model.eval()

        #print("PGD test epsilon test= " + str(eps_test) + "  num_iterTest = " + str(num_iterTest)  + "  alpha_test = " + str(alpha_test)
        t4 = time.time()

        adv_err, adv_loss, _data = self.test_epoch_adversarial(self.loader.test_loader, self.model, eps_test, num_iterTest, alpha_test)
        advTestTime = time.time() - t4

        return advTestTime, adv_err, adv_loss, _data


    def saveResults(self, pathFile, data):
        self.writeResult(pathFile, data)


    def parallelizeModel(self, model):
        if self.devices_id is not None: # parallelize the job 
            model = nn.DataParallel(model, device_ids = self.devices_id)
        return model.to(self.device)
            

    def resetModel(self, ):
        dropout_rate=0.5

        if self.dataset_name == "imageNet":
            num_classes = 1000
            depth = 50
            dropout_rate = 0.3

            # IMPORTANT (ImageNet memory fix):
            # Use the standard torchvision ResNet stem (stride-2 conv + maxpool) by default.
            # Our custom ResNet implementation is CIFAR-style (no early downsampling),
            # which can explode activation memory on 224x224 inputs and OOM even on H100.
            use_torchvision = os.environ.get("IMAGENET_TORCHVISION", "1").lower() in ("1","true","yes","y")

            if use_torchvision:
                # Use torchvision ResNet with the standard ImageNet stem (stride-2 conv + maxpool)
                if depth == 18:
                    ctor = tv_models.resnet18
                elif depth == 34:
                    ctor = tv_models.resnet34
                elif depth == 50:
                    ctor = tv_models.resnet50
                elif depth == 101:
                    ctor = tv_models.resnet101
                else:
                    ctor = tv_models.resnet50

                # torchvision API compatibility (weights vs pretrained)
                try:
                    model = ctor(weights=None)
                except TypeError:
                    model = ctor(pretrained=False)

                in_features = model.fc.in_features
                # keep dropout behavior similar to your original code
                if dropout_rate is not None and float(dropout_rate) > 0:
                    model.fc = nn.Sequential(nn.Dropout(p=float(dropout_rate)), nn.Linear(in_features, num_classes))
                else:
                    model.fc = nn.Linear(in_features, num_classes)
            else:
                # Fallback to your custom implementation (may OOM on 224x224 if not ImageNet-stem)
                if depth == 18:
                    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)
                elif depth == 34:
                    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, depth, dropout_rate)
                elif depth == 50:
                    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, depth, dropout_rate)
                elif depth == 101:
                    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, depth, dropout_rate)
                else:
                    depth = 28
                    widen_factor = 10
                    dropout_rate = 0.3
                    model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes)

        elif self.dataset_name == "svhn":
            num_classes=10
            depth = 18

            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)


        elif self.dataset_name == "cifar10" or self.dataset_name == "cifar10-c":
            num_classes=10
            depth = 18

            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)
#            return PreActResNet(PreActBlock, [2,2,2,2], num_classes, dropout_rate)

        elif self.dataset_name == "binaryCifar10":
            num_classes=2
            depth = 18

            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)
#            return PreActResNet(PreActBlock, [2,2,2,2], num_classes, dropout_rate)

        elif self.dataset_name == "cifar100":
            num_classes=100
            depth = 28
            widen_factor=10
            dropout_rate=0.3
        
            model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes)

        else:
            print("CHECK the models firts")
        


        #model = resnet18(num_classes=num_classes)
        return model.to(self.device)
                    
        return model.to(self.device)


def main(ckptName, runName, dataset_name, stop_val, stop,
         stage1_epochs, stage2_epochs,
         ecg_lam, ecg_tau, ecg_k, ecg_conf_type, ecg_detach_gates,
         ecg_schedule, ecg_lam_start, ecg_lam_end, ecg_tau_start, ecg_tau_end, ecg_k_start, ecg_k_end, ecg_adapt_warmup, ecg_adapt_window,
         ecg_tau_target, ecg_tau_lr, ecg_tau_ema, ecg_tau_deadzone, ecg_tau_min, ecg_tau_max,
         device, devices_id, lr, momentum, batch, lr_adv, momentum_adv, batch_adv,
         half_prec=False, variants='none', log_runtime=True, rt_sample_every=20,
         stage2_fast=False, stage2_find_every=3, stage2_ce_log_every=5, stage2_lr_scale=0.1,
         eval_extra_every=0, eval_adv_suite=False, adv_attacks='fgsm,pgd_linf,pgd_linf_rs',
         adv_eps=8, adv_steps=20, adv_restarts=1, adv_pixel=True,
         eval_c_suite=False, c_corruptions='gaussian_noise,brightness', c_severities=5,
         imbalance='none', imb_factor=None, imb_seed=None,
         ecg_lam_max=1.5, ecg_lam_beta=0.9, ecg_lam_eps=1e-6, seed=None,
         ecg_tail_ratio_target=3.0, ecg_tail_ratio_beta=0.9, ecg_active_frac_floor=0.05,
         ecg_sparse_lam_decay=0.5, ecg_sparse_lam_zero=False,
         ecg_tail_lam_ema=0.9, ecg_tail_invalid_decay=0.95,
         ecg_gate_temp=1.5,
         args=None):
    if seed is not None:
        os.environ["TRAIN_DATALOADER_SEED"] = str(seed)  # for DataLoader worker_init_fn (ImageNet etc.)

    dataset_loader = dataset(dataset_name=dataset_name, batch_size=batch, batch_size_adv=batch_adv,
                             imbalance=imbalance, imb_factor=imb_factor, imb_seed=imb_seed, seed=seed)
    model_cnn = model(dataset_loader, dataset_name, device, devices_id, lr, momentum, lr_adv, momentum_adv, batch_adv, half_prec=half_prec, variants=variants)
    # ECG hyperparams from CLI
    model_cnn.ecg_lam = float(ecg_lam)
    model_cnn.ecg_tau = float(ecg_tau)
    model_cnn.ecg_k = float(ecg_k)
    model_cnn.ecg_conf_type = str(ecg_conf_type)
    model_cnn.ecg_gate_temp = float(ecg_gate_temp)
    model_cnn.ecg_detach_gates = bool(ecg_detach_gates)
    # Auto-lambda: ecg_lam_start="auto"|"auto_w"|"auto_d"|"auto_dw" + ecg_lam_end=delta (or initial_delta for auto_d/auto_dw).
    # auto_w = 5-epoch delta warmup; auto_d = auto-delta (reference-based); auto_dw = auto-delta + 5-epoch warmup.
    _lam_start, _lam_end = ecg_lam_start, ecg_lam_end
    _lam_rule = str(_lam_start).strip().lower() if _lam_start else None
    if _lam_rule in _AUTO_LAM_RULES:
        model_cnn.ecg_lam_rule = _lam_rule
        model_cnn.ecg_lam_delta = float(_lam_end) if (_lam_end is not None and str(_lam_end).strip()) else 0.05
        model_cnn.ecg_lam_max = float(ecg_lam_max)
        model_cnn.ecg_lam_beta = float(ecg_lam_beta)
        model_cnn.ecg_lam_eps = float(ecg_lam_eps)
        model_cnn._ecg_gate_ema = None
        if _lam_rule in ("auto_d", "auto_dw"):
            model_cnn._ecg_delta_cur = float(model_cnn.ecg_lam_delta)
            model_cnn._ecg_delta_eff = float(model_cnn.ecg_lam_delta)
            model_cnn._ecg_scale_p99_ema = None
            model_cnn.ecg_auto_d_target_p99 = 1.55
            model_cnn.ecg_auto_d_beta_p99 = 0.9
            model_cnn.ecg_auto_d_eta = 0.05
            model_cnn.ecg_auto_d_delta_min = 0.01
            model_cnn.ecg_auto_d_delta_max = 0.20
            model_cnn.ecg_auto_d_warmup_epochs = 5
        if _lam_rule in ("auto_tr", "auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
            model_cnn.ecg_tail_ratio_target = float(ecg_tail_ratio_target)
            model_cnn.ecg_tail_ratio_beta = float(ecg_tail_ratio_beta)
            model_cnn.ecg_active_frac_floor = float(ecg_active_frac_floor)
            model_cnn.ecg_sparse_lam_decay = float(ecg_sparse_lam_decay)
            model_cnn.ecg_sparse_lam_zero = bool(ecg_sparse_lam_zero)
            model_cnn.ecg_tail_lam_ema = float(ecg_tail_lam_ema)
            model_cnn.ecg_tail_invalid_decay = float(ecg_tail_invalid_decay)
            model_cnn._ecg_gate_p95_ema = None
            model_cnn._ecg_gate_p99_ema = None
            model_cnn._ecg_active_frac_ema = None
            model_cnn._ecg_lam_auto_tr_ema = None
            if _lam_rule in ("auto_tr_sustain", "auto_tr_autocap", "auto_tr_autocap_gate"):
                model_cnn._ecg_scale_p99_ema_tr = None
                model_cnn._ecg_lam_sustain_ema = 0.0
            if _lam_rule in ("auto_tr_autocap", "auto_tr_autocap_gate"):
                _lm = float(ecg_lam_max)
                model_cnn._ecg_lam_cap_cur = min(_lm, 1.0)
                model_cnn._ecg_lam_cap_min = min(0.8, _lm)
                model_cnn._ecg_lam_cap_hit_ema = 0.0
            if _lam_rule == "auto_tr_autocap_gate":
                model_cnn._ecg_gate_q_correction = 0.0
        _lam_start, _lam_end = None, None
    else:
        model_cnn.ecg_lam_rule = getattr(model_cnn, "ecg_lam_rule", None)
        _lam_start = float(_lam_start) if (_lam_start is not None and str(_lam_start).strip()) else None
        _lam_end = float(_lam_end) if (_lam_end is not None and str(_lam_end).strip()) else None
    # Tau: auto_q (scheduled quantile q), auto_q_ctrl (P-controller q),
    #      auto_q_valley (valley-detection q), quantile (fixed q), or numeric
    # auto_q:        ecg_tau_start=auto_q,        ecg_tau_end=q_start or q_start_q_end (e.g. 0.6 or 0.6_0.85)
    # auto_q_ctrl:   ecg_tau_start=auto_q_ctrl,   ecg_tau_end=frac_start_frac_target (e.g. 0.64_0.10)
    #                Both values are active gate fractions (fraction of samples above tau threshold).
    #                frac_start -> q_start = 1 - frac_start (initial quantile).
    #                frac_target = target active fraction the P controller converges to.
    #                Controller params reuse: ecg_tau_lr (step, def 0.05), ecg_tau_ema (smoothing, def 0.9),
    #                ecg_tau_deadzone (dead zone, def 0.02), ecg_tau_min/max (q bounds, def 0.1/0.99).
    # auto_q_valley: ecg_tau_start=auto_q_valley, ecg_tau_end=q_start (optional, default 0.6)
    #                Detects valley in confidence distribution each epoch; q adapts automatically.
    #                No frac_target needed. Params: ecg_tau_valley_warmup (epochs before adapting, def 5),
    #                ecg_tau_valley_smooth (Gaussian kernel width in bins, def 3), ecg_tau_ema (def 0.9),
    #                ecg_tau_min/max (q bounds, def 0.1/0.99).
    _tau_start, _tau_end = ecg_tau_start, ecg_tau_end
    if _tau_start is not None and str(_tau_start).strip().lower() == "auto_q":
        model_cnn.ecg_tau_rule = "auto_q"
        _tau_end_str = str(_tau_end).strip() if _tau_end is not None else ""
        if "_" in _tau_end_str:
            _qs, _qe = _tau_end_str.split("_", 1)
            model_cnn.ecg_tau_q_start = float(_qs)
            model_cnn.ecg_tau_q_end = float(_qe)
        else:
            model_cnn.ecg_tau_q_start = float(_tau_end_str) if _tau_end_str else 0.6
            _auto_q_end_default = 0.7 if str(ecg_conf_type).strip().lower() == "log_pmax" else 0.9
            model_cnn.ecg_tau_q_end = _auto_q_end_default
        model_cnn.ecg_tau_quantile_cur = float(model_cnn.ecg_tau_q_start)
        _tau_start, _tau_end = None, None
    elif _tau_start is not None and str(_tau_start).strip().lower() == "auto_q_ctrl":
        model_cnn.ecg_tau_rule = "auto_q_ctrl"
        _tau_end_str = str(_tau_end).strip() if _tau_end is not None else ""
        if "_" in _tau_end_str:
            _fs, _ft = _tau_end_str.split("_", 1)
            _frac_start = float(_fs)
            model_cnn.ecg_tau_q_start       = 1.0 - _frac_start   # convert fraction -> quantile
            model_cnn.ecg_tau_q_ctrl_target = float(_ft)           # target active gate fraction
        else:
            _frac_start = float(_tau_end_str) if _tau_end_str else 0.4
            model_cnn.ecg_tau_q_start       = 1.0 - _frac_start   # default frac_start=0.4 -> q=0.6
            model_cnn.ecg_tau_q_ctrl_target = 0.1                  # default: equiv. to old q_end=0.9
        model_cnn.ecg_tau_quantile_cur = float(model_cnn.ecg_tau_q_start)
        _tau_start, _tau_end = None, None
    elif _tau_start is not None and str(_tau_start).strip().lower() == "auto_q_valley":
        model_cnn.ecg_tau_rule = "auto_q_valley"
        _tau_end_str = str(_tau_end).strip() if _tau_end is not None else ""
        model_cnn.ecg_tau_q_start = float(_tau_end_str) if _tau_end_str else 0.6
        model_cnn.ecg_tau_quantile_cur = float(model_cnn.ecg_tau_q_start)
        _tau_start, _tau_end = None, None
    elif _tau_start is not None and str(_tau_start).strip().lower() in ("quantile", "q"):
        model_cnn.ecg_tau_rule = "quantile"
        model_cnn.ecg_tau_quantile = float(_tau_end) if (_tau_end is not None and str(_tau_end).strip()) else 0.8
        _tau_start, _tau_end = None, None
    else:
        model_cnn.ecg_tau_rule = getattr(model_cnn, "ecg_tau_rule", None)
        _tau_start = float(_tau_start) if (_tau_start is not None and str(_tau_start).strip()) else None
        _tau_end = float(_tau_end) if (_tau_end is not None and str(_tau_end).strip()) else None
    # ECG schedule (full training)
    try:
        total_epochs = int(stage1_epochs) + int(stage2_epochs)
    except Exception:
        total_epochs = None
    model_cnn.configure_ecg_schedule(
        schedule=ecg_schedule,
        total_epochs=total_epochs,
        lam_start=_lam_start,
        lam_end=_lam_end,
        tau_start=_tau_start,
        tau_end=_tau_end,
        k_start=ecg_k_start,
        k_end=ecg_k_end,
        adapt_warmup=ecg_adapt_warmup,
        adapt_window=ecg_adapt_window,
        tau_target=ecg_tau_target,
        tau_lr=ecg_tau_lr,
        tau_ema=ecg_tau_ema,
        tau_deadzone=ecg_tau_deadzone,
        tau_min=ecg_tau_min,
        tau_max=ecg_tau_max,
    )
    
    model_cnn.stage1_epochs = int(stage1_epochs)
    model_cnn.stage2_epochs = int(stage2_epochs)
    # Ensure stage-1 uses the selected loss (CE by default, ECG if --loss_stage1=ecg/--full_ecg)
    model_cnn.LossInUse = LOSS_1st_stage
    model_cnn.new_iterations = int(stage2_epochs)

    model_cnn.use_wandb = (wandb.run is not None)
    model_cnn.new_iterations = int(stage2_epochs)


    # runtime diagnostics config
    model_cnn.log_runtime = bool(log_runtime)
    model_cnn.rt_sample_every = int(rt_sample_every)
    model_cnn.rt_step_sample_every = int(getattr(args, "rt_step_sample_every", 10))
    model_cnn.rt_minimal_mode = bool(getattr(args, "rt_minimal_mode", False))

    model_cnn.stage2_lr_scale = float(stage2_lr_scale)
    # stage2 speed: reduce full-train passes for large datasets (e.g. ImageNet32)
    model_cnn.stage2_fast = bool(stage2_fast)
    model_cnn.stage2_find_every = int(stage2_find_every)
    model_cnn.stage2_ce_log_every = int(stage2_ce_log_every)

    # RunA/RunB extra evals: ADV suite, C-suite, LT metrics (logged as ADV/, C/, LT/ in W&B)
    model_cnn.eval_extra_every = int(eval_extra_every)
    model_cnn.eval_adv_suite = bool(eval_adv_suite)
    model_cnn.adv_attacks = str(adv_attacks)
    model_cnn.adv_eps = float(adv_eps)
    model_cnn.adv_steps = int(adv_steps)
    model_cnn.adv_restarts = int(adv_restarts)
    model_cnn.adv_pixel = bool(adv_pixel)
    model_cnn.eval_c_suite = bool(eval_c_suite)
    model_cnn.c_corruptions = str(c_corruptions)
    model_cnn.c_severities = int(c_severities)
    model_cnn.imbalance = str(imbalance)
    model_cnn.imb_factor = float(imb_factor) if imb_factor is not None else None

    # ---- Robust training + focal/clue params ----
    model_cnn.train_mode = str(getattr(args, "train_mode", "standard"))
    model_cnn.focal_gamma = float(getattr(args, "focal_gamma", 2.0))
    model_cnn.focal_alpha = float(getattr(args, "focal_alpha", 1.0))
    model_cnn.clue_lambda = float(getattr(args, "clue_lambda", 0.2))
    model_cnn.clue_detach_proxy = bool(getattr(args, "clue_detach_proxy", True))
    model_cnn.clue_alpha = float(getattr(args, "clue_alpha", 0.5))
    model_cnn.clue_mc_passes = int(getattr(args, "clue_mc_passes", 5))
    model_cnn.clue_dropout_p = float(getattr(args, "clue_dropout_p", 0.3))
    model_cnn.clue_enable_mcdo = bool(getattr(args, "clue_enable_mcdo", True))

    _r_eps = float(getattr(args, "robust_eps", 8.0))
    _r_alpha = float(getattr(args, "robust_alpha", 0.0))
    _r_pixel = bool(getattr(args, "robust_pixel", True))
    if _r_pixel:
        _r_eps = _r_eps / 255.0
        _r_alpha = (_r_alpha / 255.0) if _r_alpha > 0 else 0.0
    _r_steps = int(getattr(args, "robust_steps", 10))
    if _r_alpha <= 0:
        _r_alpha = _r_eps / 4.0 if _r_steps <= 10 else (2.0 * _r_eps / _r_steps)
    model_cnn.robust_eps = _r_eps
    model_cnn.robust_alpha = _r_alpha
    model_cnn.robust_steps = _r_steps
    model_cnn.robust_beta = float(getattr(args, "robust_beta", 6.0))
    model_cnn.robust_random_start = bool(getattr(args, "robust_random_start", True))

    model_cnn.run(runName, iterations=int(stage1_epochs), stop=stop, ckptName=ckptName, runName=runName)
    return


def str2bool(v):
    if isinstance(v, bool):
       return v
    if str(v).lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--type', type=str, help='type of traing', default="std", choices=['std', 'robust'])
    parser.add_argument('--alg', type=str, help='path to store the model', default="pgd", choices=['pgd', 'fgsm', 'pgd_rs', 'fgsm_rs', 'fgsm_free', 'fgsm_grad_align'])

    parser.add_argument('--ratio_adv', type=float, help='percentage of data to train adversarial just for std+adv', default=1.0)
    parser.add_argument('--ratio', type=float, help='percentage of data to train adversarial', default=1.0)
    parser.add_argument('--epsilon', type=float, help='epsilon bound', default=0.1)

    parser.add_argument('--num_iter', type=int, help='number of iterations for pgd ', default=10)
    parser.add_argument('--alpha', type=float, help='alpha', default=0.01)
    
    parser.add_argument('--dataset', type=str, help='dataset', default="mnist", choices=['mnist', 'cifar10', 'cifar10-c', 'binaryCifar10', 'cifar100', 'imageNet', 'svhn'])

    parser.add_argument('--stop', type=str, help='stop condition', default="epochs", choices=['epochs', 'time'])
    parser.add_argument('--stop_val', type=int, help='number of epochs or training time', default=10)

    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.0)
    parser.add_argument('--batch', type=int, help='batch', default=100)

    parser.add_argument('--lr_adv', type=float, help='learning rate adv training', default=0.01)
    parser.add_argument('--momentum_adv', type=float, help='momentum adv training', default=0.0)
    parser.add_argument('--batch_adv', type=int, help='batch adv training', default=100)

    parser.add_argument('--workers', type=str, help='GPU workers', default="0")
    parser.add_argument('--half_prec', type=str2bool, help='half precision', default=False)

    parser.add_argument('--variants', type=str, help='calibration, deup, ensemble, cals', default='none')

    parser.add_argument(
        "--pe_mode",
        type=str,
        default="raw",
        choices=["raw", "logk", "logk_rms", "none"],
        help="PE mode: raw/none (baseline), logk (PE/logK), logk_rms (PE/logK + RMS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )


    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable stricter deterministic behavior (may reduce speed or raise errors if an op is non-deterministic).",
    )
    # ---- 2-stage loss control (CE pretrain -> method finetune) ----
    parser.add_argument("--stage1_epochs", type=int, default=30)
    parser.add_argument("--stage2_epochs", type=int, default=30)
    parser.add_argument(
        "--loss_stage1",
        type=str,
        default="ce",
        choices=["ce", "ecg"],
    )

    parser.add_argument(
        "--full_ecg",
        action="store_true",
        help="Run ECG from epoch 1 to stop_val (sets stage1_epochs=stop_val, stage2_epochs=0, loss_stage1=ecg).",
    )

    parser.add_argument(
        "--loss_stage2",
        type=str,
        default="ce",
        choices=["ce", "euat", "ecg", "ecg_abl", "focal", "clue_lite", "clue"],
    )

    # ---- CLUE params ----
    parser.add_argument("--clue_dropout_p", type=float, default=0.3)
    parser.add_argument("--clue_mc_passes", type=int, default=5)
    parser.add_argument("--clue_alpha", type=float, default=0.5,
                        help="CLUE: alpha * CE + (1-alpha) * (CE-u)^2")
    parser.add_argument("--clue_enable_mcdo", type=str2bool, default=True)

    # ---- Robust training mode (pgd_at / trades / mart) ----
    parser.add_argument("--train_mode", type=str, default="standard",
                        choices=["standard", "pgd_at", "trades", "mart"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=1.0)
    parser.add_argument("--clue_lambda", type=float, default=0.2)
    parser.add_argument("--clue_detach_proxy", type=str2bool, default=True)
    parser.add_argument("--robust_eps", type=float, default=8.0,
                        help="Adversarial eps for robust training (pixel units if robust_pixel)")
    parser.add_argument("--robust_alpha", type=float, default=0.0,
                        help="PGD step size (0=auto)")
    parser.add_argument("--robust_steps", type=int, default=10)
    parser.add_argument("--robust_beta", type=float, default=6.0,
                        help="Regularization weight for TRADES/MART")
    parser.add_argument("--robust_random_start", type=str2bool, default=True)
    parser.add_argument("--robust_pixel", type=str2bool, default=True,
                        help="If True, robust_eps/alpha are in pixel [0,255] scale")

    # ---- ECG params ----
    parser.add_argument("--ecg_lam", type=float, default=1.0)
    parser.add_argument("--ecg_tau", type=float, default=0.7)
    parser.add_argument("--ecg_k", type=float, default=10.0)
    parser.add_argument("--ecg_conf_type", type=str, default="pmax", choices=["pmax", "pmax_temp", "log_pmax", "margin", "1-pe", "logit_gap_norm", "none"])
    parser.add_argument("--ecg_gate_temp", type=float, default=1.5, help="Temperature for pmax_temp conf gate (only used when ecg_conf_type=pmax_temp).")
    parser.add_argument("--ecg_detach_gates", type=str2bool, default=True)
    parser.add_argument("--ecg_schedule", type=str, default="none", choices=["none", "linear", "cosine", "adaptive", "tau_target"])
    parser.add_argument("--ecg_lam_start", type=str, default=None,
                        help="Start lam, 'auto'/'auto_w' (auto-lambda), or 'auto_d'/'auto_dw' (auto-lambda + reference-based auto-delta). Then ecg_lam_end=delta or initial_delta.")
    parser.add_argument("--ecg_lam_end", type=str, default=None,
                        help="End lam, or delta (target pre-norm strength) when ecg_lam_start=auto.")
    parser.add_argument("--ecg_lam_max", type=float, default=1.5, help="Max lam when using auto-lambda.")
    parser.add_argument("--ecg_lam_beta", type=float, default=0.9, help="EMA beta for gate_mean in auto-lambda.")
    parser.add_argument("--ecg_lam_eps", type=float, default=1e-6, help="Eps in lam = delta/(gate_ema+eps) for auto-lambda.")
    parser.add_argument("--ecg_tau_start", type=str, default=None,
                        help="Start tau, 'quantile'/'q' (fixed q), 'auto_q' (scheduled q), or 'auto_q_ctrl' (P-controller q). "
                             "For quantile: ecg_tau_end=q. "
                             "For auto_q: ecg_tau_end=q_start (q_end defaults to 0.9) or q_start_q_end (e.g. 0.6_0.85). "
                             "For auto_q_ctrl: ecg_tau_end=frac_start_frac_target (e.g. 0.64_0.10); both are active gate fractions. "
                             "frac_start is converted to q_start=1-frac_start internally. frac_target is the controller convergence target. "
                             "For auto_q_valley: ecg_tau_end=q_start (optional, default 0.6); q adapts each epoch by detecting the valley "
                             "in the confidence histogram. No frac_target needed. Extra params: ecg_tau_valley_warmup (def 5), "
                             "ecg_tau_valley_smooth (def 3), ecg_tau_ema (def 0.9), ecg_tau_min/max (def 0.1/0.99). "
                             "Controller params: ecg_tau_lr (step size, def 0.05), ecg_tau_ema (EMA smoothing, def 0.9), "
                             "ecg_tau_deadzone (dead zone, def 0.02), ecg_tau_min/max (q bounds, def 0.1/0.99).")
    parser.add_argument("--ecg_tau_end", type=str, default=None,
                        help="End tau, or quantile value (e.g. 0.8) when ecg_tau_start=quantile.")
    parser.add_argument("--ecg_k_start", type=float, default=None)
    parser.add_argument("--ecg_k_end", type=float, default=None)

    parser.add_argument("--ecg_adapt_warmup", type=int, default=10)
    parser.add_argument("--ecg_adapt_window", type=int, default=5)
    # ---- tau_target schedule (Scheme C): adapt tau to keep gate active fraction near a target ----
    parser.add_argument("--ecg_tau_target", type=float, default=0.6,
                        help="Target fraction of samples with conf>tau (i.e., active gates) when --ecg_schedule=tau_target.")
    parser.add_argument("--ecg_tau_lr", type=float, default=0.10,
                        help="Update step for tau_target: tau <- tau + lr*(active_frac-target).")
    parser.add_argument("--ecg_tau_ema", type=float, default=0.90,
                        help="EMA beta for active_frac (0 disables EMA). Higher = smoother.")
    parser.add_argument("--ecg_tau_deadzone", type=float, default=0.02,
                        help="Deadzone for tau_target updates. If |active_frac-target|<deadzone, do not update tau.")
    parser.add_argument("--ecg_tau_min", type=float, default=0.0)
    parser.add_argument("--ecg_tau_max", type=float, default=0.99)
    parser.add_argument("--ecg_tau_valley_warmup", type=int, default=5,
                        help="Epochs before auto_q_valley starts adapting q (default 5).")
    parser.add_argument("--ecg_tau_valley_smooth", type=int, default=3,
                        help="Gaussian kernel half-width in bins for confidence histogram smoothing in auto_q_valley (default 3).")
    # ---- tail-ratio auto-lambda (auto_tr) ----
    parser.add_argument("--ecg_tail_ratio_target", type=float, default=3.0,
                        help="Target tail amplification ratio (1+lam*g_p99)/(1+lam*g_mean) for auto_tr.")
    parser.add_argument("--ecg_tail_ratio_beta", type=float, default=0.9,
                        help="EMA beta for gate_p99 and active_frac tracking in auto_tr.")
    parser.add_argument("--ecg_active_frac_floor", type=float, default=0.05,
                        help="Sparse-gate guard: if active_frac_ema < this, reduce lambda.")
    parser.add_argument("--ecg_sparse_lam_decay", type=float, default=0.5,
                        help="Multiply lambda by this factor when sparse-gate guard fires.")
    parser.add_argument("--ecg_sparse_lam_zero", type=str2bool, default=False,
                        help="If True, set lambda=0 (instead of decay) when sparse-gate guard fires.")
    parser.add_argument("--ecg_tail_lam_ema", type=float, default=0.9,
                        help="EMA beta for smoothing lambda over time in auto_tr (0=no smoothing).")
    parser.add_argument("--ecg_tail_invalid_decay", type=float, default=0.95,
                        help="When tail-ratio denom is invalid, decay previous smoothed lam by this factor.")

    # ---- runtime diagnostics ----
    parser.add_argument("--log_runtime", type=str2bool, default=True)
    parser.add_argument("--rt_sample_every", type=int, default=20,
                        help="Measure loss-call latency every N calls (CUDA uses events + 1 sync per epoch).")
    parser.add_argument("--rt_step_sample_every", type=int, default=10,
                        help="Measure full training step latency every N batches.")
    parser.add_argument("--rt_minimal_mode", type=str2bool, default=False,
                        help="If True, skip optional loss-call timing diagnostics (keeps step+wall timing).")

    # ---- eval / adv suite / C-suite / imbalance / dump (used by run_from_tsv sweeps) ----
    parser.add_argument("--eval_extra_every", type=int, default=0, help="Run extra eval every N epochs (0 = off).")
    parser.add_argument("--eval_adv_suite", type=str2bool, default=False, help="Run adversarial eval suite.")
    parser.add_argument("--adv_attacks", type=str, default="fgsm,pgd_linf,pgd_linf_rs", help="Comma-separated list of adv attacks.")
    parser.add_argument("--adv_eps", type=float, default=8, help="Epsilon for adv attacks (pixel scale).")
    parser.add_argument("--adv_steps", type=int, default=20, help="Steps for iterative adv attacks.")
    parser.add_argument("--adv_restarts", type=int, default=1, help="Restarts for adv attacks.")
    parser.add_argument("--adv_alpha", type=float, default=None, help="Step size for adv (default derived from eps/steps).")
    parser.add_argument("--adv_pixel", type=str2bool, default=True, help="Adv epsilon in pixel scale (vs 0-1).")
    parser.add_argument("--eval_c_suite", type=str2bool, default=False, help="Run C (corruption) eval suite.")
    parser.add_argument("--c_corruptions", type=str, default="gaussian_noise,brightness", help="Comma-separated C corruptions.")
    parser.add_argument("--c_severities", type=int, default=5, help="Severity level for C suite.")
    parser.add_argument("--c_name", type=str, default="", help="Single C corruption name (overrides c_corruptions if set).")
    parser.add_argument("--c_severity", type=int, default=None, help="Single C severity (overrides c_severities if set).")
    parser.add_argument("--imbalance", type=str, default="none", help="Class imbalance: none | longtail | ...")
    parser.add_argument("--imb_factor", type=float, default=None, help="Imbalance factor for long-tail.")
    parser.add_argument("--imb_seed", type=int, default=None, help="Seed for imbalance sampling.")
    parser.add_argument("--dump_gates", type=str2bool, default=False, help="Dump gate stats for demo.")
    parser.add_argument("--dump_gates_n", type=int, default=2000, help="Number of samples for dump_gates.")

    parser.add_argument("--stage2_lr_scale", type=float, default=0.1,
                        help="Multiply lr by this when entering stage2 (avoids representation collapse, e.g. PGD 0.5).")
    # ---- stage2 speed (reduce full passes for large datasets e.g. ImageNet32) ----
    parser.add_argument("--stage2_fast", type=str2bool, default=False,
                        help="Reduce stage2 full passes: find misclassified every N epochs, log train CE every M.")
    parser.add_argument("--stage2_find_every", type=int, default=3,
                        help="When stage2_fast: run full-train 'find wrong' pass every this many epochs (reuse split otherwise).")
    parser.add_argument("--stage2_ce_log_every", type=int, default=5,
                        help="When stage2_fast: run compute_train_ce_err every this many epochs (reuse last for W&B otherwise).")

    parser.add_argument("--force_run", action="store_true")

    args = parser.parse_args()

    def _apply_stage2_from_args(args):
        global LOSS_2nd_stage_wrong, LOSS_2nd_stage_correct, option_stage2
        if args.loss_stage2 == "ce":
            LOSS_2nd_stage_wrong = LOSS_MIN_CROSSENT
            LOSS_2nd_stage_correct = LOSS_MIN_CROSSENT
            option_stage2 = "batch_mix2"
        elif args.loss_stage2 == "euat":
            LOSS_2nd_stage_wrong = LOSS_MIN_CROSSENT_MAX_UNC
            LOSS_2nd_stage_correct = LOSS_MIN_CROSSENT
            option_stage2 = "batch_mix2"
        elif args.loss_stage2 in ("ecg", "ecg_abl"):
            LOSS_2nd_stage_wrong = LOSS_ECG
            LOSS_2nd_stage_correct = LOSS_ECG
            option_stage2 = "batch_mix2"
        elif args.loss_stage2 == "focal":
            LOSS_2nd_stage_wrong = LOSS_FOCAL
            LOSS_2nd_stage_correct = LOSS_FOCAL
            option_stage2 = "batch_mix2"
        elif args.loss_stage2 == "clue_lite":
            LOSS_2nd_stage_wrong = LOSS_CLUE_LITE
            LOSS_2nd_stage_correct = LOSS_CLUE_LITE
            option_stage2 = "batch_mix2"
        elif args.loss_stage2 == "clue":
            LOSS_2nd_stage_wrong = LOSS_CLUE
            LOSS_2nd_stage_correct = LOSS_CLUE
            option_stage2 = "batch_mix2"
        else:
            raise ValueError(f"Unknown --loss_stage2: {args.loss_stage2}")

    _apply_stage2_from_args(args)
    print(f"[CFG] LOSS2 wrong={LOSS_2nd_stage_wrong}, correct={LOSS_2nd_stage_correct}, option={option_stage2}")

def set_seed(seed: int, deterministic: bool = False):
    """
    Set seeds for python/numpy/torch. If deterministic=True, enable stricter deterministic settings.

    Notes:
    - deterministic=True can slow training and may raise errors if a used op is inherently non-deterministic.
    - For maximum determinism on CUDA, also export:
        CUBLAS_WORKSPACE_CONFIG=:4096:8
        PYTHONHASHSEED=<seed>
    """
    import os, random, numpy as np, torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Always keep these two for more stable results across runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        # Optional: reduce TF32 nondeterminism / numeric drift
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        # Best-effort: ensure deterministic algorithms (may error on unsupported ops)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[WARN] torch.use_deterministic_algorithms(True) failed: {e}")

        # Best-effort: cublas workspace config (works best if set before process starts)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def init_wandb_if_needed(args, default_name: str):
    import os, wandb
    project = os.environ.get("WANDB_PROJECT", "ecg_binary_fast") #os.environ.get("WANDB_PROJECT", "ecg_binary_fast")
    entity  = os.environ.get("WANDB_ENTITY", None)
    name    = os.environ.get("WANDB_NAME", None) or default_name
    group   = os.environ.get("WANDB_GROUP", None) or f"{args.dataset}_seed{args.seed}_T{args.stop_val}"

    tags_env = os.environ.get("WANDB_TAGS", "")
    tags = [t.strip() for t in tags_env.split(",") if t.strip()]

    base_tags = {
        args.dataset, f"seed{args.seed}", f"s1{args.stage1_epochs}", f"s2{args.stage2_epochs}",
        f"loss2_{args.loss_stage2}", f"pe_{args.pe_mode}",
    }
    tags = sorted(set(tags) | base_tags)
    # wandb allows tags between 1 and 64 chars
    WANDB_MAX_TAG_LEN = 64
    tags = [t[:WANDB_MAX_TAG_LEN] if len(t) > WANDB_MAX_TAG_LEN else t for t in tags]

    wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=tags,
        job_type=os.environ.get("WANDB_JOB_TYPE", None),
        config=vars(args),
    )
    # Make charts use "epoch" as x-axis by default (instead of Step)
    try:
        wandb.define_metric("epoch")
        for prefix in ("train/*", "STD/*", "PGD/*", "ECG/*", "ADV/*", "C/*", "LT/*", "TIME/*", "MEM/*", "config/*"):
            wandb.define_metric(prefix, step_metric="epoch")
    except Exception:
        pass

    try:
        if wandb.run is not None and torch.cuda.is_available():
            wandb.log({"MEM/device_name": torch.cuda.get_device_name(0)}, step=0)
    except Exception:
        pass

    global PE_MODE, Normalize_entropy, USE_PE_RMS
    PE_MODE = args.pe_mode
    Normalize_entropy = (PE_MODE in ("logk", "logk_rms"))  # none/raw = no PE normalization
    USE_PE_RMS = (PE_MODE == "logk_rms")
    print(f"[CFG] pe_mode={PE_MODE} Normalize_entropy={Normalize_entropy} USE_PE_RMS={USE_PE_RMS} seed={args.seed}")

if __name__ == "__main__":
    args = parser.parse_args()

    # Convenience: run ECG for the whole training
    if getattr(args, "full_ecg", False):
        args.loss_stage1 = "ecg"
        args.stage1_epochs = int(args.stop_val)
        args.stage2_epochs = 0

    # Auto-enable stage2_fast when stage2 is active: batch_mix2 runs a full forward pass
    # every mini-batch iteration when stage2_fast=False, making stage2 ~500x slower than expected.
    if int(args.stage2_epochs) > 0 and (not args.stage2_fast):
        args.stage2_fast = True
        args.stage2_find_every = max(args.stage2_find_every, 5)
        args.stage2_ce_log_every = max(args.stage2_ce_log_every, 10)

    # Apply stage-1 loss selection
    if args.loss_stage1 == "ce":
        LOSS_1st_stage = LOSS_MIN_CROSSENT
    elif args.loss_stage1 == "ecg":
        LOSS_1st_stage = LOSS_ECG
    else:
        raise ValueError(f"Unknown --loss_stage1: {args.loss_stage1}")

    _apply_stage2_from_args(args)
    print(f"[CFG] LOSS2 wrong={LOSS_2nd_stage_wrong}, correct={LOSS_2nd_stage_correct}, option={option_stage2}")

    set_seed(args.seed, deterministic=getattr(args, 'deterministic', False))

    dataset_name = args.dataset
    if args.type == "std":
        args.ratio = 0.0
        args.ratio_adv = 0.0

    modelName = "model" + dataset_name
    modelName += "_std_train" if args.type == "std" else "_robust"
    modelName += "_lr" + str(args.lr) + "_momentum" + str(args.momentum) + "_batch" + str(args.batch)
    modelName += "_lrAdv" + str(args.lr_adv) + "_momentumAdv" + str(args.momentum_adv) + "_batchAdv" + str(args.batch_adv)
    modelName += f"_pe{args.pe_mode}"

    baseName  = modelName + f"_s1{args.stage1_epochs}_{args.loss_stage1}"
    stage2Name = baseName + f"_s2{args.loss_stage2}"

    if (args.loss_stage1 == "ecg") or args.loss_stage2.startswith("ecg"):
        stage2Name += f"_conf{args.ecg_conf_type}_dg{int(args.ecg_detach_gates)}"
        if args.ecg_schedule != "none":
            stage2Name += f"_sched{args.ecg_schedule}"
            if args.ecg_schedule in ["linear", "cosine"]:
                _ls = getattr(args, "ecg_lam_start", None)
                _ls_lower = str(_ls).strip().lower() if _ls else ""
                if _ls_lower in _AUTO_LAM_RULES:
                    lam_part = f"lam_{_ls_lower}{args.ecg_lam_end or '0.05'}"
                else:
                    lam_s = _ls if _ls is not None else args.ecg_lam
                    lam_e = args.ecg_lam_end if args.ecg_lam_end is not None else args.ecg_lam
                    lam_part = f"lam{lam_s}-{lam_e}"
                _ts = getattr(args, "ecg_tau_start", None)
                _ts_lower = str(_ts).strip().lower() if _ts else ""
                if _ts_lower in ("quantile", "q"):
                    tau_part = f"tauq{args.ecg_tau_end or '0.8'}"
                elif _ts_lower == "auto_q":
                    _aq = str(args.ecg_tau_end or '0.6').replace("_", "-")
                    tau_part = f"tau_autoq{_aq}"
                else:
                    tau_s = _ts if _ts is not None else args.ecg_tau
                    tau_e = args.ecg_tau_end if args.ecg_tau_end is not None else args.ecg_tau
                    tau_part = f"tau{tau_s}-{tau_e}"
                k_s = args.ecg_k_start if args.ecg_k_start is not None else args.ecg_k
                k_e = args.ecg_k_end if args.ecg_k_end is not None else args.ecg_k
                stage2Name += f"_{lam_part}_{tau_part}_k{k_s}-{k_e}"
            elif args.ecg_schedule == "adaptive":
                stage2Name += f"_warm{args.ecg_adapt_warmup}_win{args.ecg_adapt_window}_baseLam{args.ecg_lam}_baseTau{args.ecg_tau}_baseK{args.ecg_k}"
        else:
            _ls = getattr(args, "ecg_lam_start", None)
            _ls_lower = str(_ls).strip().lower() if _ls else ""
            lam_disp = f"lam_{_ls_lower}{args.ecg_lam_end or '0.05'}" if _ls_lower in ("auto", "auto_w", "auto_d", "auto_dw") else f"lam{args.ecg_lam}"
            _ts = getattr(args, "ecg_tau_start", None)
            _ts_lower = str(_ts).strip().lower() if _ts else ""
            if _ts_lower in ("quantile", "q"):
                tau_disp = f"tauq{args.ecg_tau_end or '0.8'}"
            elif _ts_lower == "auto_q":
                _aq = str(args.ecg_tau_end or '0.6').replace("_", "-")
                tau_disp = f"tau_autoq{_aq}"
            else:
                tau_disp = f"tau{args.ecg_tau}"
            stage2Name += f"_{lam_disp}_{tau_disp}_k{args.ecg_k}"

    if args.variants == "cals":
        baseName += "_cals"
        stage2Name += "_cals"

    if int(args.stage2_epochs) <= 0:
        stage2Name = baseName

    runName = stage2Name

    import torch, os, sys
    devices_id = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckptName = baseName
    runName  = stage2Name

    toRun = args.force_run or (not os.path.isfile("./logs/logs_" + runName + ".txt"))
    if not toRun:
        print("config exists (no checkpointing is needed)")
        sys.exit(0)

    init_wandb_if_needed(args, default_name=runName)

    # Log varying hyperparams at step=0 so Wandb charts can "Color by" / "Group by" and show separate run curves
    try:
        import wandb
        if wandb.run is not None:
            lam_s = getattr(args, "ecg_lam_start", None)
            tau_s = getattr(args, "ecg_tau_start", None)
            k_s = getattr(args, "ecg_k_start", None)
            _lam_start_val = 0.0
            _lam_delta = None
            if lam_s is not None and str(lam_s).strip().lower() in ("auto", "auto_w", "auto_d", "auto_dw"):
                _lam_delta = float(getattr(args, "ecg_lam_end", None) or 0.05)
            else:
                _lam_start_val = float(lam_s) if lam_s is not None else 0.0
            _tau_start_val = None
            _tau_q = None
            if tau_s is not None and str(tau_s).strip().lower() in ("quantile", "q"):
                _tau_start_val = 0.0  # placeholder; use ecg_tau_quantile for actual
                _tau_q = float(getattr(args, "ecg_tau_end", None) or 0.8)
            elif tau_s is not None and str(tau_s).strip().lower() == "auto_q":
                _tau_start_val = 0.0  # placeholder; q_start in ecg_tau_end
                _tau_q = float(getattr(args, "ecg_tau_end", None) or 0.6)  # q_start
            else:
                _tau_start_val = float(tau_s) if tau_s is not None else 0.0
            to_log = {"epoch": 0, "config/ecg_lam_start": _lam_start_val,
                      "config/ecg_tau_start": _tau_start_val, "config/ecg_k_start": float(k_s) if k_s is not None else 0.0}
            if _lam_delta is not None:
                to_log["config/ecg_lam_delta"] = _lam_delta
            if _tau_q is not None:
                to_log["config/ecg_tau_quantile"] = _tau_q
            wandb.log(to_log, step=0)
    except Exception:
        pass

    main(
        ckptName, runName, dataset_name,
        args.stop_val, args.stop,
        args.stage1_epochs, args.stage2_epochs,
        args.ecg_lam, args.ecg_tau, args.ecg_k, args.ecg_conf_type, args.ecg_detach_gates,
        args.ecg_schedule, args.ecg_lam_start, args.ecg_lam_end, args.ecg_tau_start, args.ecg_tau_end, args.ecg_k_start, args.ecg_k_end, args.ecg_adapt_warmup, args.ecg_adapt_window,
        args.ecg_tau_target, args.ecg_tau_lr, args.ecg_tau_ema, args.ecg_tau_deadzone, args.ecg_tau_min, args.ecg_tau_max,
        device, devices_id,
        args.lr, args.momentum, args.batch,
        args.lr_adv, args.momentum_adv, args.batch_adv,
        args.half_prec, args.variants,
        args.log_runtime, args.rt_sample_every,
        args.stage2_fast, args.stage2_find_every, args.stage2_ce_log_every, getattr(args, "stage2_lr_scale", 0.1),
        getattr(args, "eval_extra_every", 0), getattr(args, "eval_adv_suite", False),
        getattr(args, "adv_attacks", "fgsm,pgd_linf,pgd_linf_rs"), getattr(args, "adv_eps", 8),
        getattr(args, "adv_steps", 20), getattr(args, "adv_restarts", 1), getattr(args, "adv_pixel", True),
        getattr(args, "eval_c_suite", False), getattr(args, "c_corruptions", "gaussian_noise,brightness"),
        getattr(args, "c_severities", 5), getattr(args, "imbalance", "none"), getattr(args, "imb_factor", None),
        getattr(args, "imb_seed", None),
        getattr(args, "ecg_lam_max", 1.5), getattr(args, "ecg_lam_beta", 0.9), getattr(args, "ecg_lam_eps", 1e-6),
        getattr(args, "seed", 0),
        getattr(args, "ecg_tail_ratio_target", 3.0), getattr(args, "ecg_tail_ratio_beta", 0.9),
        getattr(args, "ecg_active_frac_floor", 0.05), getattr(args, "ecg_sparse_lam_decay", 0.5),
        getattr(args, "ecg_sparse_lam_zero", False),
        getattr(args, "ecg_tail_lam_ema", 0.9), getattr(args, "ecg_tail_invalid_decay", 0.95),
        getattr(args, "ecg_gate_temp", 1.5),
        args=args,
    )