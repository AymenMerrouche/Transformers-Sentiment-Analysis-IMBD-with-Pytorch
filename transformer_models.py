import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from torch.optim import Adam
import numpy as np
from utils import *
import math

class MLP(nn.Module):
    """
        A dense network :  feedforward neural network with relu activations
    """
    def __init__(self, input_size, output_size, layers = []):
        """
        :param input_size: MLP input size
        :param output_size: MLP output size
        :param layers: Layers of the MLP
        """
        super(MLP, self).__init__()
        self.layers = layers
        self.output_size = output_size
        if len(layers)>0:
            fc = [nn.Linear(input_size, self.layers[0], bias = False)]
            fc.append(nn.ReLU())
            for i in range(len(self.layers)-1):
                fc.append(nn.Linear(self.layers[i], self.layers[i+1], bias = False))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(self.layers[-1], output_size, bias = False))
        else:
            fc = [nn.Linear(input_size, output_size, bias = False)]
        self.linear = nn.Sequential(*fc)
    
    def forward(self, x):
        return self.linear(x)
    
    
class GlobalAveragePooling(nn.Module):
    """
    Represents an input as the mean of its embeddings, and pass it
    to a simple linear layer.
    """
    def __init__(self, embedding_size, num_classes=2):
        """
            :param embedding_size: size of embedding
            :param num_classes: number of classes
        """

        super().__init__()
        self.embedding_size = embedding_size

        self.lin = nn.Linear(embedding_size, num_classes)

    def forward(self, x, lengths):
        x = x.transpose(1,0)
        mean = torch.sum(x, 0) / lengths.unsqueeze(1)
        out = self.lin(mean)
        return out

class SelfAttention(nn.Module):
    """
        Self Attention module 
    """
    def __init__(self, embedding_size, attention_heads=8, layers_q = [], layers_k = [], layers_v = [], attention_type = "wide"):
        """
        :param embedding_size: size of embeddings
        :param attention_heads: number of attention heads
        :param layers_q: layers of the projection to compute query
        :param layers_k: layers of the projection to compute key
        :param layers_v: layers of the projection to compute value
        :param attention_type: type of attention (wide of narrow)
        """

        super().__init__()

        self.embedding_size = embedding_size
        self.attention_heads = attention_heads
        self.attention_type = attention_type

        
        
        if self.attention_type == "wide":
            
            # compute keys, values and queries for every attention head
            self.get_keys = MLP(embedding_size, embedding_size * attention_heads, layers_k)
            #nn.Linear(embedding_size, embedding_size * attention_heads, bias=False)
            self.get_queries = MLP(embedding_size, embedding_size * attention_heads, layers_q)
            #nn.Linear(embedding_size, embedding_size * attention_heads, bias=False)
            self.get_values = MLP(embedding_size, embedding_size * attention_heads, layers_v)
            #nn.Linear(embedding_size, embedding_size * attention_heads, bias=False)
            
            # merge the attention heads in the wide attention fashion
            self.merge_attention_heads = nn.Linear(attention_heads * embedding_size, embedding_size, bias=False)

    def forward(self, x, mask):
        
        b, t, e = x.size() # number of sequences per batch * number of items per sequence * embedding size
        
        h = self.attention_heads # number of attention heads
        
        # Self attention operations (Weight matrices and then the output sequence)
        
        # compute keys, queries and values
        keys    = self.get_keys(x)   .view(b, t, h, e)
        queries = self.get_queries(x).view(b, t, h, e)
        values  = self.get_values(x) .view(b, t, h, e)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        
        # - get dot product of queries and keys, and scale
        w = torch.bmm(queries, keys.transpose(1, 2))
        
        # scale by sqrt(embedding size = e)
        w = w / math.sqrt(e) 
        
        # some of the sequences are of length inferior to the fixed sequence size, the correponding words (flagged in mask)
        # They should not be attended for (assign zero weights to padding i.e. exp(-inf) = 0 in the softmax)
        w = w.masked_fill(mask.unsqueeze(-1).repeat(self.attention_heads, 1, t).transpose(1, 2),-float('Inf'))
        # soft max to get values in [0, 1] that sum to 1 for each embeddinng
        w = F.softmax(w, dim=2) # to [0, 1] using softmax

        # apply the self attention to the values
        output_sequence = torch.bmm(w, values).view(b, h, t, e)

        # swap h, t back, marge the attention heads
        output_sequence = output_sequence.transpose(1, 2).contiguous().view(b, t, h * e)
        if self.attention_heads>1:
            return self.merge_attention_heads(output_sequence)
        else:
            return output_sequence
    

class Transformer(nn.Module):
    """
        Transformer Block
    """
    def __init__(self,  embedding_size, attention_heads=8, n_parallel_ffd=4, dropout=0.0):
        """
        :param embedding_size: size of input embeddings
        :param attention_heads: number of attention heads in self attention
        :param n_parallel_ffd: number of parallel ffnn after attention
        :param dropout: dropout parameter
        """
        super().__init__()
        
        # one self attention layer
        self.attention = SelfAttention(embedding_size, attention_heads)

        # norm layers
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # feed forward layers
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, n_parallel_ffd * embedding_size),
            nn.ReLU(),
            nn.Linear(n_parallel_ffd * embedding_size, embedding_size)
        )        
        # Dropout
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask):

        # self attend
        attention_result = self.attention(x, mask)
        
        # g_theta function
        
        # norm attention result + residual
        x = self.norm1(attention_result + x)
        # dropout 
        x = self.do(x)
        # compute feedfoward
        fedforward = self.ff(x)
        # norm feedfroward result + residual
        x = self.norm2(fedforward + x)
        # dropout
        x = self.do(x)

        return x
        
class ClassificationWithTransformers(nn.Module):
    """
        Stack of tranformers, take the mean of the result and project it to number_of_classes with a linear projection (global average pooling)
    """
    
    def __init__(self,  embedding_size, attention_heads=8, n_parallel_ffd=4, dropout=0.0, L=3, num_classes=2, pos_enc = None):
        """

        :param embedding_size: size of input embeddings
        :param attention_heads: number of attention heads
        :param n_parallel_ffd: number of parallel MLPs that process the result of self Attention in the transformers
        :param dropout: dropout parameter used in the transformer block
        :param L: size of the transformer stack
        :param num_classes: number of classes of the classification problem
        """

        super().__init__()
        self.L= L
        
        # positional encoding
        self.pos_enc = pos_enc
        
        # the transformer stack
        self.transformer_stack = nn.ModuleList([Transformer(embedding_size, attention_heads, n_parallel_ffd, dropout) for i in range(L-1)])
        
        # Average pooling
        self.average_pooling = GlobalAveragePooling(embedding_size, num_classes)
        
    def forward(self, x, mask, lengths):
        
        # add positional encodings if available
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        
        # get the result of the transformer stack
        transformer_stack_result = self.transformer_stack[0](x, mask)
        for l in range (1, self.L-1):
            # note that we mask similarily to the first transformer
            # that is because a transformer returns tokens of same size as input (seq to seq)
            transformer_stack_result = self.transformer_stack[0](transformer_stack_result, mask)
            
        # Global Average Pooling : avergae the stansformer stack result and project to number_of_classes
        result = self.average_pooling(transformer_stack_result, lengths)
        
        return result
    
    
class ClassificationWithTransformers_PCLS(nn.Module):
    """
        Stack of tranformers, take the learned cls embedding and project it to number_of_classes with a linear projection
    """
    
    def __init__(self,  embedding_size, attention_heads=8, n_parallel_ffd=4, dropout=0.0, L=3, num_classes=2, pos_enc = None):
        """

        :param embedding_size: size of input embeddings
        :param attention_heads: number of attention heads
        :param n_parallel_ffd: number of parallel MLPs that process the result of self Attention in the transformers
        :param dropout: dropout parameter used in the transformer block
        :param L: size of the transformer stack
        :param num_classes: number of classes of the classification problem
        """

        super().__init__()
        self.L= L
        
        # pseudo token CLS
        self.cls_embedding = torch.randn(1, embedding_size) # initialize according to gaussian distribution (i.e. as if this is the embedding of [CLS])
        self.cls = nn.Linear(embedding_size, embedding_size, bias = False) # use a linear projection to learn it
        
        # positional encoding
        self.pos_enc = pos_enc
        
        # the transformer stack
        self.transformer_stack = nn.ModuleList([Transformer(embedding_size, attention_heads, n_parallel_ffd, dropout) for i in range(L-1)])
        
        # Average pooling
        self.classify = nn.Linear(embedding_size, num_classes)
        
        
    def forward(self, x, mask, lengths):

        # create cls embedding and pass it to a linear_layer(embedding_size, embedding_size) to learn it
        cls_embedding_repeted = self.cls_embedding.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device) # same embedding for each sequence
        cls_embedding_learned = self.cls(cls_embedding_repeted)
        
        # add the learned cls embeddings in the beggining of each sequence
        x_cls = torch.cat((cls_embedding_learned, x), dim=1)

        # mask for the cls token
        cls_mask = torch.full((x.size(0), 1), False, dtype = torch.bool).to(mask.device)
        mask = torch.cat((cls_mask, mask), dim=1)

        # add positional encodings if available
        if self.pos_enc is not None:
            x_cls = self.pos_enc(x_cls)

        # get the result of the transformer stack
        transformer_stack_result = self.transformer_stack[0](x_cls, mask)
        for l in range (1, self.L-1):
            # note that we mask similarily to the first transformer
            # that is because a transformer returns tokens of same size as input (seq to seq)
            transformer_stack_result = self.transformer_stack[0](transformer_stack_result, mask)

        # Project the contextualised CLS token in num_classes
        result = self.classify(transformer_stack_result[:,0,:])
        
        return result