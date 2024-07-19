# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:38:56 2023

@author: spisak
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import pytorch_lightning as pl

from dataset import makeDataset
import torch.nn.functional as F

########## originally proposed in: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim#

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        if self.dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0,1,0,0))
        return emb
    
###### Many classes originally from: https://github.com/tcapelle/Diffusion-Models-pytorch
class   DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None,batch_size=1,label_frames=444,device="cuda", **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.labelFrames = label_frames
        self.num_classes = num_classes
        self.time_dim = time_dim
        if num_classes is not None:
            self.sinu_pos_emb = SinusoidalPosEmb(time_dim)
            self.label_emb = nn.Embedding(num_classes, time_dim)
            
        self.inc = DoubleConv(c_in,64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 128)
        self.sa2 = SelfAttention(128)
        self.device = device
        self.incy = DoubleConv(c_in,64)
        self.down1y = Down(64, 128)
        self.sa1y = SelfAttention(128)
        self.down2y = Down(128, 128)
        self.sa2y = SelfAttention(128)

        self.bot1 = DoubleConv(128, 128)
        self.bot2 = DoubleConv(128, 128)

        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0.0, channels, 2, device = self.device))
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    def mask(self,x):
        start_x = torch.randint(high=54,size=(1,))
        start_y = torch.randint(high=54,size=(1,))
        x[start_x:start_x+10,start_y:start_y+10] = 0
        return x
    def unet_forwad(self, x, t, y):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        y1 = self.incy(y)
        y2 = self.down1y(y1, t)
        y2 = self.sa1y(y2)
        y3 = self.down2y(y2, t)
        y3 = self.sa2y(y3)
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        x = self.up2(x3, y2, t)
        x = self.sa5(x)
        x = self.up3(x, y1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    def forward(self, x, t, y):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t, y)
        
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

class litUNet(pl.LightningModule):
    def __init__(self,loss_weights=[1,1,1], c_in=3, c_out=3, time_dim=256, label_frames=444, num_classes=2,device="cuda",dataset=None):
        super(litUNet,self).__init__()
        self.noise_steps  = 1000
        self.dataset = dataset
        self.beta_start= 1e-4
        self.lossWeights = loss_weights
        self.beta_end = 2e-2
        self.beta =cosine_beta_schedule(1000).float()        
        self.alpha = 1. -self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)
        self.classWeight = torch.tensor([1,1])
        self.ce = nn.CrossEntropyLoss()
        self.in_channels = c_in
        self.out_channels=c_out
        self.embed_dim = time_dim
        self.time_dim = time_dim
        self.batch_size =1
        self.epochLoss = 0
        self.labelFrames = label_frames
        self.numClasses = num_classes
        self.model = UNet_conditional(c_in = self.in_channels,
                                      c_out=self.out_channels,
                                      num_classes=self.numClasses,
                                      batch_size = self.batch_size,
                                      label_frames=self.labelFrames,
                                      device=device)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.counter = 0
        self.correct= 0
        self.valCorrect = 0
        self.validationSteps = 0
        self.bce = nn.BCELoss()
        self.bestValAcc = 0
        self.valAcc = 0
        self.trainSteps = 0
        self.bestTrainAcc = 0
        self.trainAcc = 0
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps,requires_grad=False).to(self.device)
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,),requires_grad=False)
    def forward(self, x, t,y):
        output= self.model(x,t.float(),y)
        return output
    def noise_images(self, x, t):
        self.alpha_hat = self.alpha_hat.to(x.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(self.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(self.device)
        e = torch.randn_like(x).to(x.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e
    def training_step(self, batch, batch_idx):
        goal, condition = batch
        t = self.sample_timesteps(goal.shape[0]).to(self.device)
        encoded_labels, noise = self.noise_images(goal, t)
        predicted_noise = self(encoded_labels,t,condition)
        loss = self.mse(predicted_noise,goal)
        self.epochLoss+=loss.item()
        self.trainSteps +=1
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    def train_dataloader(self):
        training_data,_ = makeDataset(self.dataset,arms=False)
        train_loader = DataLoader(training_data, num_workers=6, batch_size=self.batch_size, shuffle=True)
        return train_loader
    def on_train_epoch_end(self):
        print("Loss:", self.epochLoss/self.trainSteps)
        self.epochLoss = 0
        self.trainSteps = 0


if __name__ == "__main__":
    myModel = litUNet([1,0,0],dataset="twoPerspectivesData.csv")
    trainer = pl.Trainer( max_epochs=2,devices=1,fast_dev_run=True,log_every_n_steps=1)
    trainer.fit(model=myModel)
