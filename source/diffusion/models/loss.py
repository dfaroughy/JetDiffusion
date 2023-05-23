
import torch
import torch.nn as nn
import numpy as np

def calculate_loss(model, sde, data, args, loss_func=None, reduction=torch.mean):
    if not loss_func: loss_func = args.loss
    loss = reduction(loss_func(model, sde, data, args))
    return loss

def denoising_loss(model, sde, batch, args, tol=1e-5):
    ''' denoising score matching loss
    '''
    batch = batch[:, :args.dim] 
    t = torch.rand(batch.shape[0]) * (sde.T - tol) + tol  
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(t)
    batch_noisy =  batch + z * std 
    t,  std , batch_noisy= t.to(args.device), std.to(args.device), batch_noisy.to(args.device)
    score = model(batch_noisy, t)
    loss = torch.square(score * std + batch_noisy)
    return loss
