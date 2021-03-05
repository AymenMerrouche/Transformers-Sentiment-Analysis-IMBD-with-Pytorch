import itertools
import logging
from tqdm import tqdm
from pathlib import Path
import os
import yaml
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from typing import List
import torch.nn as nn
import time
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



class CheckpointState():
    """A model checkpoint state."""
    def __init__(self, model, optimizer=None, epoch=1, savepath='./checkpt.pt'):

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.savepath = Path(savepath)

    def state_dict(self):
        """Checkpoint's state dict, to save and load"""
        dict_ = dict()
        dict_['model'] = self.model.state_dict()
        if self.optimizer:
            dict_['optimizer'] = self.optimizer.state_dict()
        dict_['epoch'] = self.epoch
        return dict_

    def save(self, suffix=''):
        """Serializes the checkpoint.
        Args:
            suffix (str): if provided, a suffix will be prepended before the extension
                of the object's savepath attribute.
        """
        if suffix:
            savepath = self.savepath.parent / Path(self.savepath.stem + suffix +
                                                   self.savepath.suffix)
        else:
            savepath = self.savepath
        with savepath.open('wb') as fp:
            torch.save(self.state_dict(), fp)

    def load(self):
        """Deserializes and map the checkpoint to the available device."""
        with self.savepath.open('rb') as fp:
            state_dict = torch.load(
                fp, map_location=torch.device('cuda' if torch.cuda.is_available()
                                              else 'cpu'))
            self.update(state_dict)

    def update(self, state_dict):
        """Updates the object with a dictionary
        Args:
            state_dict (dict): a dictionary with keys:
                - 'model' containing a state dict for the checkpoint's model
                - 'optimizer' containing a state for the checkpoint's optimizer
                  (optional)
                - 'epoch' containing the associated epoch number
        """
        self.model.load_state_dict(state_dict['model'])
        if self.optimizer is not None and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']
        


