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
from diffusion.models.diffusion.deep_architectures import ScoreNet
from diffusion.models.diffusion.dynamics import driftlessSDE
from diffusion.models.loss import denoising_loss

sys.path.append("../")
torch.set_default_dtype(torch.float64)

#########################################

params = argparse.ArgumentParser()
params.add_argument('--dir', type=str)

#########################################

if __name__ == '__main__':

    #...get model params
    args = params.parse_args()
    make_dir(args.dir+'/plots', overwrite=True)
    with open(args.dir + '/inputs.json', 'r') as f: model_inputs = json.load(f)
    args = argparse.Namespace(**model_inputs)

    args.num_time_steps=1000


    #...get datasets

    file =  "./data/events_anomalydetection_v2.features_with_jet_constituents.h5"
    data = torch.tensor(pd.read_hdf(file).to_numpy())
    bckg = torch.cat((data[:, :4], data[:, 7:11], data[:, -2:]), dim=1)  # d=9: (jet1, jet2, mjj, truth_label)
    bckg = shuffle(bckg)

    #...get SB events and preprocess data

    bckg = EventTransform(bckg, args)
    # bckg.compute_mjj()

    #...define template model

    sde = driftlessSDE(args)
    score = ScoreNet(args, marginal_prob=sde.marginal_prob)
    model = Model(score, sde, args) 

    #...load model

    model.load_state(path=args.workdir+'/best_score_model.pth')

    #...define template model

    sample = model.sample(num_samples=bckg.num_jets, num_batches=10)
    sample = torch.cat((sample, torch.zeros(sample.shape[0],1)), dim=1)
    sample = EventTransform(sample, args, convert_to_ptepm=False)
    sample.mean = torch.tensor(args.mean)
    sample.std = torch.tensor(args.std)
    sample.min = torch.tensor(args.min)
    sample.max = torch.tensor(args.max)
    sample.preprocess(reverse=True)
    sample.compute_mjj()

    jet_plot_routine((sample.data, bckg.data), 
                     title='jet features generated SR ', save_dir=args.workdir+'/plots')




