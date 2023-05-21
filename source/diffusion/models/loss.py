
import torch
import torch.nn as nn
import numpy as np

def calculate_loss(model, data, args, loss_func=None, reduction=torch.mean):
    if not loss_func: loss_func = args.loss
    loss = reduction(loss_func(model, data, args))
    return loss

def denoising_loss(model, sde, batch, args, tol=1e-5):
    ''' denoising scroe matching loss
    '''
    batch = batch[:, :args.dim]
    batch = batch.to(args.device)
    t = torch.rand(batch.shape[0], device=args.device) * (1. - tol) + tol  
    noise = torch.randn_like(batch)
    mean, std = sde.marginal_prob(t)
    batch_noisy =  batch + noise * std[:, None, None, None]
    score = model(batch_noisy, t)
    loss = torch.sum((score * std[:, None, None, None] + batch_noisy)**2, dim=(1,2,3))
    return loss
