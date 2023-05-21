import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import copy
from copy import deepcopy
import argparse
import json
import h5py
import pandas as pd

from diffusion.utils.base import make_dir, copy_parser, save_arguments, shuffle
from diffusion.data.transform import EventTransform 
from diffusion.data.plots import jet_plot_routine

from diffusion.models.training import Model
from diffusion.models.diffusion.architectures import MlpScore
from diffusion.models.diffusion.dynamics import driftlessSDE
from diffusion.models.loss import denoising_loss


sys.path.append("../")
torch.set_default_dtype(torch.float64)


'''
    Description:

    Normalizing flow (maf or coupling layer) for learning the 
    features of the dijet distribution for lhco data. 

    The flow generates context for a diffusion model for anomaly detection.

 
'''

####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the flow model')

params.add_argument('--device',       default='cuda:1',           help='where to train')
params.add_argument('--dim',          default=8,                  help='dim of data: (pT1,eta1,phi1,m1,pT2,eta2,phi2,m2)', type=int)
params.add_argument('--loss',         default=denoising_loss,     help='loss function')
params.add_argument('--sde',          default='driftless',        help='type of dynamics: driftless, VPSDE, VESDE, sunVPSDE', type=str)
params.add_argument('--N',            default=1000,               help='number of time steps' , type=int)
params.add_argument('--dim_embed',    default=128,               help='dim of time embedding' , type=int)

#...flow params:


#...training params:

params.add_argument('--batch_size',    default=1024,         help='size of training/testing batch', type=int)
params.add_argument('--num_steps',     default=0,            help='split batch into n_steps sub-batches + gradient accumulation', type=int)
params.add_argument('--test_size',     default=0.2,          help='fraction of testing data', type=float)
params.add_argument('--max_epochs',    default=2000 ,        help='max num of training epochs', type=int)
params.add_argument('--max_patience',  default=20,           help='terminate if test loss is not changing', type=int)
params.add_argument('--lr',            default=1e-4,         help='learning rate of generator optimizer', type=float)
params.add_argument('--seed',          default=999,          help='random seed for data split', type=int)

#... data params:

params.add_argument('--mass_window', default=(0,3300,3700,13000), help='bump hunt mass window: SB1, SR, SB2', type=tuple)

####################################################################################################################

if __name__ == '__main__':

    #...create working folders 

    args = params.parse_args()
    args.workdir = make_dir('Results_dijet_density', overwrite=True)

    #...get datasets

    file =  "./data/events_anomalydetection_v2.features_with_jet_constituents.h5"
    data = torch.tensor(pd.read_hdf(file).to_numpy())
    data = torch.cat((data[:, :4], data[:, 7:11], data[:, -2:]), dim=1)  # d=9: (jet1, jet2, mjj, truth_label)
    data = shuffle(data)


    #...get SB events and preprocess data

    side_bands = EventTransform(data, args)
    side_bands.compute_mjj()
    side_bands.get_sidebands()
    side_bands.preprocess()

    #...store parser arguments

    args.num_jets = side_bands.num_jets
    args.num_gen = side_bands.num_jets
    args.mean = side_bands.mean.tolist()
    args.std = side_bands.std.tolist()
    args.max = side_bands.max.tolist()
    args.min = side_bands.min.tolist()
    print("INFO: num sideband jets: {}".format(args.num_jets))
    save_arguments(args, name='inputs.json')

    # #...Prepare train/test samples from sidebands

    train, test  = train_test_split(side_bands.data, test_size=args.test_size, random_state=args.seed)

    # #...define model

    sde = driftlessSDE(args)
    score = MlpScore(args, marginal_prob_std=sde.marginal_prob)
    model = Model(score, sde, args) 

    #...train flow for density estimation.

    train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args.batch_size, shuffle=True)
    test_sample  = DataLoader(dataset=torch.Tensor(test),  batch_size=args.batch_size, shuffle=False) 
    
    model.train(train_sample, test_sample)
    