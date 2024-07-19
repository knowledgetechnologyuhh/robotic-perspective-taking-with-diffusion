# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:38:56 2023

@author: spisak
"""
#%%
from einops import rearrange
from functools import partial
from tqdm.auto import tqdm
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import makeDataset
from main import litUNet

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

######## 
######## from https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class diffusion_Inference():
    
    def __init__(self, model,
                 condition,
                 timesteps = 1000,
        sampling_timesteps = 50,
        schedule_fn_kwargs = dict(),
        objective = 'pred_x0',
        ddim_sampling_eta = .1,):
        self.device = "cuda"
        self.timesteps = timesteps
        self.beta_start= 1e-4
        self.beta_end = 2e-2
        self.model = model
        self.betas = cosine_beta_schedule(1000).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_recip_alphas_cumprod =  torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod =  torch.sqrt(1. / self.alphas_cumprod-1)

        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.objective = objective
        self.self_condition = condition
        self.unnormalize = unnormalize_to_zero_to_one 
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, 1000)
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (torch.sqrt(self.alphas_cumprod[t])[:, None,None ,None ]* x_t - x0) / \
            torch.sqrt(1 - self.alphas_cumprod[t])[:, None,None,None]
            )
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -noise )/
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) 
        )
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        if type(x_self_cond) != list:
            x_self_cond = x_self_cond.cuda()
        model_output = self.model(x.float().cuda(), t.cuda(), x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
         pred_noise = model_output
         x_start = self.predict_start_from_noise(x, t, pred_noise)
         x_start = maybe_clip(x_start)
 
         if clip_x_start and rederive_pred_noise:
             pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)


        return (pred_noise, x_start)
    

    
    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        _batch, device, total_timesteps, sampling_timesteps, eta, _objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)  
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 

        x = torch.randn(shape, device = device)

        for time, time_next in time_pairs:
            
            t = (torch.ones(1)*time).long().to(self.device)
            pred_noise, x_start= self.model_predictions(x, t, self.self_condition, clip_x_start = False, rederive_pred_noise = True)
            x = x_start
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

    
            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if time_next < 0:
                x = x_start
                continue
        return x
def pixelWiseAccuarcy(prediction,label):
    count = 0
    correct = 0
    for predictedPixels, labelPixels in zip(prediction,label):
        for predictedPixel,labelPixel in zip(predictedPixels,labelPixels):
            count+=1
            if predictedPixel.all() == labelPixel.all():
                correct+=1
    return correct/count

def inferImgToIMG():
    myModel = litUNet()
    myModel.load_state_dict(torch.load("./trainedModelCos.pt"))
    td_,td = makeDataset("twoPerspectivesData.csv",arms=False)
    tl = DataLoader(td,batch_size=1,shuffle=True)
    it = iter(tl)
    myModel.cuda()
    myModel.eval()
    goal, condition = next(it)
    noise = torch.randn((1,3,64,64))
    t =  torch.Tensor([999]).int()
    directPred = myModel(noise.cuda(),t.cuda(),condition.cuda())
    diffusion = diffusion_Inference(myModel,condition)

    ret = diffusion.ddim_sample((1,3,64,64))
    ret = rearrange(ret.cpu().detach().numpy().squeeze(), "c h w -> h w c")
    goal = rearrange(goal.cpu().detach().numpy().squeeze(), "c h w -> h w c")

    plt.title("goal")
    plt.axis("off")
    plt.imshow(goal)
    
    plt.show()
    condition  = rearrange(condition.cpu().detach().numpy().squeeze(), "c h w -> h w c")
    plt.title("condition")
    plt.axis("off")
    plt.imshow(condition)
    plt.show()

    plt.title("ret")
    plt.axis("off")
    plt.imshow(ret)
    plt.show()

    directPred = rearrange(directPred.cpu().detach().numpy().squeeze(), "c h w -> h w c")
    
    plt.title("pred")
    plt.axis("off")
    plt.imshow(directPred)
    plt.show()

if __name__ == "__main__":
    inferImgToIMG()