from data_utils import *
from utils import *
from train import *
from models import *
from transformer_models import *
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical


# Load args

# general args
with open('./configs/general_parameters.yaml', 'r') as stream:
    gen_args  = yaml.load(stream,Loader=yaml.Loader)
# Transformer Classification args
with open('./configs/transformer_classification_parameters.yaml', 'r') as stream:
    classif_args  = yaml.load(stream,Loader=yaml.Loader)
    
# load the data  

word2id, embeddings, train_loader, test_loader = get_dataloaders(gen_args["embedding_size"], gen_args["batch_size"], gen_args["max_length"])
embeddings = torch.as_tensor(embeddings, dtype=torch.float).to(device)

# positional encodings
pos_enc  = None
if classif_args["pos"]:
    if classif_args["cls"]:
        pos_enc = PositionalEncoding(gen_args["embedding_size"], gen_args["max_length"]+1)
    else :
        pos_enc = PositionalEncoding(gen_args["embedding_size"], gen_args["max_length"])
    


# the network
if classif_args["cls"] :
    net = ClassificationWithTransformers_PCLS(gen_args["embedding_size"], classif_args["attention_heads"], classif_args["n_parallel_ffd"], classif_args["dropout"], pos_enc = pos_enc).to(device)
else :
    net = ClassificationWithTransformers(gen_args["embedding_size"], classif_args["attention_heads"], classif_args["n_parallel_ffd"], classif_args["dropout"], pos_enc = pos_enc).to(device)

    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=gen_args['lr'])

# Logging + Experiment

ignore_keys = {'no_tensorboard'}
# get hyperparameters with values in a dict
hparams = {**gen_args, **classif_args}
# generate a name for the experiment
expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
print("Experimenting with : \n \t"+expe_name)
# path where to save the model
savepath = Path('models/checkpt.pt')
# Tensorboard summary writer
if gen_args['no_tensorboard']:
    writer = None
else:
    writer = SummaryWriter("runs/runs"+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+expe_name)
# start the experiment
checkpoint = CheckpointState(net, optimizer, savepath=savepath)
fit(checkpoint, criterion, embeddings, train_loader, test_loader, gen_args['epochs'], writer=writer)

if not gen_args['no_tensorboard']:
    writer.close()