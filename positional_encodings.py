import torch
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import PositionalEncoding
sns.set()

# where to save the plots
output_dir = "heatmaps/"

# dimension of embedding space
embedding_size = 100
# different sequence lengths to test
seq_legths = [200, 500, 1000, 2000, 3000, 4000, 5000]

# for each sequence length, generate the correpsonding heat map
for seq_length in seq_legths:
    print(seq_length)
    # toy sequence of 0 to pass to the positional encoding class
    shape = (1, seq_length, embedding_size)
    x = torch.zeros(shape)
    pos_enc = PositionalEncoding(embedding_size, seq_length)
    x = pos_enc(x)
    sns.set()
    # dot product matrix
    dt_product_matrix = torch.bmm(x,x.transpose(1,2)).squeeze(0)
    sns_plot = sns.heatmap(dt_product_matrix, cbar=False)
    sns_plot.figure.savefig(output_dir+"heatmap"+" embedding_size = "+str(embedding_size)+" sequence_legth = "+str(seq_length)+".png")
    
# positional encodings matrix
seq_length = 10
embedding_size = 20
shape = (1, seq_length, embedding_size)
pos_enc = PositionalEncoding(embedding_size, seq_length)
x = torch.zeros(shape)
x = pos_enc(x)
pos_enc_matrix = x.squeeze(0)
sns.set()
sns_plot = sns.heatmap(pos_enc_matrix, cbar=False)
plt.xlabel("profondeur k")
plt.ylabel("position i")
plt.savefig(output_dir+"heatmap_posenc"+" embedding_size = "+str(embedding_size)+" sequence_legth = "+str(seq_length)+".png")
    
