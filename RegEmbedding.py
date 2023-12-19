import os
import numpy as np
import awkward as ak
from sklearn.utils import shuffle
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import logging
import itertools
from itertools import combinations
from collections import defaultdict

from utils import pad_to
from models import HyperEmbedder

device = torch.device("cuda:0")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

batch_size = 5 # * 200 channel * 2 decays per batch 
n_files = 100 # max: 100
n_skip_files = 0
file_path = Path('/project/agkuhr/users/boyang/data/graFEI/')
n_features = 4 # features = ['px', 'py', 'pz', 'energy']
pdg_emb = 4 # embdedding size of pdg ids # Input size to NN: n_features + pdg_emb + dim_hyper
num_pdg = 13
dim_hyper = 64 # hyperbolic output: [batch, dim_hyper]

# Discriminator 
# transformer
tr_width = 256
tr_n_head = 16 # Transformer heads
tr_n = 6 # repeat Transformer blocks
tr_hidden_size = 1024 # Transformer fc dim

amplifier = 1
#r = [1,10,1] # amplifier for [intra_loss, inter_loss, radius_loss]
r = [10,1] # amplifier for [reg_loss, radius_loss]

epsilon = 1e-6 # must higher than 1e-8, important for torch.acos 
out_dir = '/home/b/Boyang.Yu/graFEI/'

model_name_hst = f'reg_htr_{dim_hyper}d'
# model_name_link = 'pretrained_Linker'

continue_training = True
freeze_train = False

logging.basicConfig(
    filename=f'{out_dir}log_{model_name_hst}.log', 
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
    )
# training parameters
lr = 1e-6
patience = 5
delta = lr/100

logging.info(f'countinue: {continue_training}, frozen: {freeze_train}')
logging.info(f'file {n_skip_files}-{n_skip_files+n_files-1}')
logging.info(f'lr: {lr}')

class Dataset(Dataset):    
    def __init__(self, arrays, batch_size=1, shuffle=True):
        if shuffle:
            random_idx = np.random.permutation(len(arrays))
            self.arrays = arrays[random_idx]
        else:
            self.arrays = arrays
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.arrays) // self.batch_size
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = start + self.batch_size
        events = ak.concatenate(self.arrays[start:stop])
        x = pad_to(events.pairs[:,0])
        return {
            "pdg": x[:,:,0].astype(np.int32), 
            "feature": x[:,:,1:], 
            "padding_mask": x[:,:,0]>0,
            "mass": events.masses.to_numpy().astype(np.int32), 
            "pattern": events.patterns.to_numpy().astype(np.int32), 
            "links": pad_to(events.links,fill_value=-1,dtype=np.int_),
            "channels": events.channels.to_numpy().astype(np.int32), 
            "evtNums": events.evtNums.to_numpy().astype(np.int32)
        }
    
def build_loader(files, batch_size=1):
    dataloaders = []
    n_batchs = 0
    for file in files:
        array = ak.from_parquet(file)
        ds = Dataset(array, batch_size)
        n_batchs += len(ds)
        dataloaders.append(torch.utils.data.DataLoader(ds, batch_size=None))
    return itertools.chain.from_iterable(dataloaders), n_batchs

batched = defaultdict(list)
for key in ["train", "val"]:
    files = sorted([file for file in file_path.glob(f'{key}_*.parquet')])[n_skip_files:n_skip_files+n_files]
    DataLoader, n_batchs = build_loader(files, batch_size=1)
    batched[key] = [batch for batch in DataLoader]
    print(f'Number of {key} batches: {len(batched[key])}')

def build_angle_matrix(vset, amplifier=None):
    M_a = vset.unsqueeze(1).repeat(1,len(vset),1)
    M_b = vset.unsqueeze(0).repeat(len(vset),1,1)
    M_ab = F.cosine_similarity(M_a, M_b, dim=-1, eps=epsilon)
    if not amplifier:
        return torch.exp(M_ab)
    else:
        return torch.exp(M_ab/amplifier), torch.exp(M_ab*amplifier)
    
def build_distance_matrix(vset):
    v = 1 / (1 - torch.norm(vset, dim=1)**2)
    D = torch.norm(vset[:,None]-vset, dim=2)**2
    return torch.exp(-torch.acosh(1 + 2*v*D*v[:,None] + epsilon))

def radius_loss(vectors, dataset):
    r_euclidean = torch.norm(vectors, dim=-1)**2
    r_poincare = torch.acosh(1+2*r_euclidean/(1-r_euclidean-epsilon))
    r_goal = 0.6 * torch.sqrt(1-dataset['mass'].float().to(device)/100) + 0.3
    return F.mse_loss(input=r_poincare, target=r_goal)
    # return F.mse_loss(input=r_euclidean, target=r_goal)
    
def intra_loss(vectors, dataset):
    A_nurf, A_buff = build_angle_matrix(vectors, amplifier=amplifier)
    mask = (dataset['evtNums'][:,None]==dataset['evtNums']).to(device)
    norm = torch.einsum('ij,ij->j',A_nurf,(~mask).float()) + epsilon
    return (-torch.log(A_buff/norm)[mask]).mean()

def inter_loss(vectors, dataset):
    v_pattern = dataset["pattern"].float()
    M_a = v_pattern.unsqueeze(1).repeat(1,len(v_pattern),1)
    M_b = v_pattern.unsqueeze(0).repeat(len(v_pattern),1,1)
    M_ab = F.cosine_similarity(M_a, M_b, dim=-1, eps=epsilon).to(device)
    A_angle = build_angle_matrix(vectors)
    norm_angle = A_angle.sum(dim=-1)
    A_distance = build_distance_matrix(vectors)
    norm_distance = A_distance.sum(dim=-1)
    return (M_ab*(-torch.log(A_angle/norm_angle) - torch.log(A_distance/norm_distance))).mean()

def VICReg_loss(vectors, dataset):
    v_pattern = dataset["pattern"].float()
    M_a = v_pattern.unsqueeze(1).repeat(1,len(v_pattern),1)
    M_b = v_pattern.unsqueeze(0).repeat(len(v_pattern),1,1)
    M_ab = F.cosine_similarity(M_a, M_b, dim=-1, eps=epsilon).to(device)
    norm_ab = M_ab.sum(dim=0).unsqueeze(-1)
    z_a = vectors.unsqueeze(1).repeat(1,len(vectors),1)
    z_b = vectors.unsqueeze(0).repeat(len(vectors),1,1)
    # invariance loss
    #sim_loss = (F.mse_loss(z_a, z_b, reduction='none') * M_ab.unsqueeze(-1)).mean()
    sim_loss = (build_distance_matrix(vectors) * M_ab).mean()
    # variance loss
    z = z_a - (torch.einsum('ijk,ij->jk', z_a, M_ab) / norm_ab).unsqueeze(0)
    std_z = torch.sqrt(torch.einsum('ijk,ij->jk', z**2, M_ab) / (norm_ab-1) + epsilon)
    std_loss = torch.mean(F.relu(1 - std_z))
    # covariance loss
    cov_z = (z.transpose(-1,-2) @ (z*M_ab.unsqueeze(-1)) - torch.eye(z.size(-1),device=device)) / (norm_ab.unsqueeze(-1)-1)
    cov_loss = cov_z.pow(2).sum() / z.size(-1)

    return sim_loss + 10 * std_loss + 10* cov_loss

model_emb = HyperEmbedder(
    n_features=n_features,
    tr_width=tr_width,
    tr_n_head=tr_n_head,
    tr_n=tr_n,
    tr_hidden_size=tr_hidden_size,
    pdg_emb=pdg_emb,
    dim_hyper=dim_hyper,
    num_pdg=num_pdg,
    device=device
)
model_dict = model_emb.state_dict()
    
# pretrained_dict = torch.load(f'{out_dir}{model_name_link}.pth', map_location=device)['model_state_dict']
# to_update = defaultdict(list)
# for i,k_v in enumerate(pretrained_dict.items()):
#     if k_v[0] in model_dict:
#         if k_v[1].shape==model_dict[k_v[0]].shape:
#             to_update[k_v[0]] = k_v[1]
#             freeze_threshold = i

if continue_training:
    checkpoint = torch.load(f'{out_dir}{model_name_hst}.pth', map_location=device)
    epoch_init = checkpoint['epoch']
    model_emb.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model parameters from {out_dir}{model_name_hst}.pth')
    print(f'Current epoch: {epoch_init}')
    logging.info('Continue training mode activated')
    logging.info(f'Loaded model parameters from {out_dir}{model_name_hst}.pth')
    logging.info(f'Initial epoch: {epoch_init}')
else:
    epoch_init = 0
    # model_dict.update(to_update)
    # model_emb.load_state_dict(model_dict)
# if freeze_train: # freeze the feature embedding part (before transformer)
#     for i,p in enumerate(model_emb.parameters()):
#         p.requires_grad = False
#         if i > freeze_threshold:
#             break
        
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model_emb.parameters()), lr=lr)

loss_history = []
val_loss_history = []
stop_count = 0
val_loss_min = 1000
updated = True
max_epochs = epoch_init + 2000
min_epochs = 5

logging.info('Training started')

for epoch in range(epoch_init, max_epochs):
    train_pred = []
    losses = []
    has_nan = False
    for ds in batched['train']:
        # loss calculation
        opt.zero_grad()
        output = model_emb(ds)
        # intra = intra_loss(output, ds)
        # inter = inter_loss(output, ds)
        reg_loss = VICReg_loss(output, ds)
        r_loss = radius_loss(output, ds)
        loss = r[0]*reg_loss + r[1]*r_loss
    
        if torch.isnan(loss.detach()):
            stop_count += 1
            updated = False
            has_nan = True
            print(intra, inter, r_loss)
            print(output)
            checkpoint = torch.load(f'{out_dir}{model_name_hst}.pth')
            epoch = checkpoint['epoch']
            model_emb.load_state_dict(checkpoint['model_state_dict'])
            val_loss_history = val_loss_history[:epoch+1]
            print('Nan detected, roll back to epoch %i'%epoch)
            logging.warning('Nan detected, roll back to epoch %i'%epoch)
            break
            
        # gradient update
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_emb.parameters(), max_norm=1, error_if_nonfinite=False)
        opt.step()
        losses.append(loss.detach().cpu().numpy())
        train_loss = np.mean(losses)
        meta_msg = f'Training epoch {epoch}, '
        #loss_msg = f'intra loss {intra}, inter loss {inter}, radius loss {r_loss}, train loss {train_loss}'
        loss_msg = f'reg loss {reg_loss}, radius loss {r_loss}, train loss {train_loss}'
        
    # epoch train summary
    loss_history.append(train_loss)

    # validation
    val_losses = []
    with torch.no_grad():
        for ds in batched['val']:
            output = model_emb(ds)
            # intra = intra_loss(output, ds)
            # inter = inter_loss(output, ds)
            reg_loss = VICReg_loss(output, ds)
            r_loss = radius_loss(output, ds)
            loss = r[0]*reg_loss + r[1]*r_loss
            val_losses.append(loss.detach().cpu().numpy()) 
            val_loss = np.mean(val_losses)
        val_loss_history.append(val_loss)
            
        # early stopping and checkpoint sl
        if epoch > min_epochs:
            if (val_loss_min - val_loss_history[-1])/abs(val_loss_min) < delta:# insufficient update
                stop_count += 1
                updated = False

            elif not has_nan:
                stop_count = 0
                val_loss_min = val_loss_history[-1]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_emb.state_dict()
                    }, f'{out_dir}{model_name_hst}.pth')
                updated = True
                
        if stop_count >= patience:                                
            # run out of patience, stop and roll back
            checkpoint = torch.load(f'{out_dir}{model_name_hst}.pth')
            epoch = checkpoint['epoch']
            print('Early stopped and roll back to epoch %i'%epoch)
            logging.warning('Early stopped and roll back to epoch %i'%epoch)            
            model_emb.load_state_dict(checkpoint['model_state_dict'])
            val_loss_history = val_loss_history[:epoch+1]
            break
        else:
            meta_msg = f'Epoch:{epoch}/{max_epochs}, '
            batch_msg = f'Loss: {val_loss_history[-1]}, '
            training_msg = f'Updated: {updated}, Patience: {stop_count}/{patience}'
            #last_batch_loss_msg = f'intra loss {intra}, inter loss {inter}, radius loss {r_loss}, '
            last_batch_loss_msg = f'reg loss {reg_loss}, radius loss {r_loss}, '
            print(meta_msg + batch_msg)
            print(training_msg)
            print('For the last batch: ' + last_batch_loss_msg)  
            logging.info(meta_msg + batch_msg)
            logging.info(training_msg)  
            logging.info('For the last batch: ' + last_batch_loss_msg)  

# plt.plot(loss_history, label='Train')
# plt.plot(val_loss_history, label='Validation')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.savefig(f"Training_{model_name_hst}.pdf")
