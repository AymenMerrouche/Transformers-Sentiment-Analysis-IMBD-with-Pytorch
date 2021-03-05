import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        seq = self.tokenizer(s if isinstance(s, str) else s.read_text())
        label = self.filelabels[ix]
        nb_tokens = min(self.max_length, len(seq))
        return seq[:nb_tokens], label
def get_imdb_data(embedding_size=50, max_length = 100):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    word2id = {word: ix for ix, word in enumerate(words)}
    words.append("__OOV__")
    words.append("__PAD__")
    
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))
    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False, max_length=max_length), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False, max_length=max_length)

def collate_fn(sequences):
    ls_array_X = []
    ls_array_Y = []
    lengths = torch.tensor([len(sequences[i][0]) for i in range(len(sequences))])
    for i in range(len(sequences)):
        ls_array_X.append(torch.tensor(sequences[i][0]))
        ls_array_Y.append(torch.tensor(sequences[i][1]))
        
    myPadX = torch.nn.utils.rnn.pad_sequence(sequences=ls_array_X, batch_first=True,padding_value=-1)
    labels = torch.tensor(ls_array_Y)
    mask = torch.arange(myPadX.size(1)).repeat(myPadX.size(0),1) >= lengths.unsqueeze(1).view(-1)[ : ,None]
    return myPadX, lengths, mask,labels

def get_dataloaders(embedding_size=50, batch_size=128, max_length=100):

    word2id, embeddings, text_train, text_test = get_imdb_data(embedding_size, max_length)
    train_loader = DataLoader(text_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(text_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return word2id, embeddings, train_loader, test_loader


class PositionalEncoding(nn.Module):
    "Position embeddings"

    def __init__(self, d_model: int, max_len: int = 5000):
        """Génère des embeddings de position

        Args:
            d_model (int): Dimension des embeddings à générer
            max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Ajoute les embeddings de position"""
        x = 2*x + self.pe[:, :x.size(1)].repeat(x.shape[0],1,1)
        return x