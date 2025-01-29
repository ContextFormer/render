import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

import copy


class linear_prob(nn.Module):
    def __init__(self,in_dim, out_dim):
        super().__init__()
        self.layer=nn.Linear(in_dim, out_dim)

    def forward(self,fea):
        return self.layer(fea)

class unlinear_predictor(nn.Module):
    def __init__(self,in_dim,emb_dim,out_dim):
        super().__init__()
        self.latent_linear=nn.Linear(in_dim+emb_dim,out_dim)

    def forward(self,input,z):
        print(input.view(1,1).size(),z.size())
        return self.latent_linear(torch.cat([input.view(1,1).float(),z.float()],dim=-1))

if __name__=='__main__':

    sequence_length =100

    data_range=np.linspace(0,1,sequence_length)

    y1=1*data_range
    y2=copy.deepcopy(y1)
    y1[0:10]=0
    y1[20:30]=0
    y1[-50:]=0
    y2[0:50]=0
    y2[60:70]=0
    y2[-10:]=0
    print(y1)
    y3=np.linspace(0,1,sequence_length)

    y3[:20]=0
    y3[-20:]=0

    y1=torch.from_numpy(y1)
    y2=torch.from_numpy(y2)
    y3=torch.from_numpy(y3)

    z1 = torch.ones(size=(1, 10))
    z2 = torch.ones(size=(1, 10))
    z3 = torch.ones(size=(1, 10))

    z1.requires_grad=True
    z2.requires_grad=True
    z3.requires_grad=True

    optimizer_z1 = torch.optim.Adam([z1], lr=0.001)
    optimizer_z2 = torch.optim.Adam([z2], lr=0.001)
    optimizer_z3 = torch.optim.Adam([z3], lr=0.001)

    linear_model=linear_prob(100,10)

    for i in range (100):
        emb1 = torch.nn.AdaptiveAvgPool1d(10)(y1.view(1,100))
        emb2 = torch.nn.AdaptiveAvgPool1d(10)(y2.view(1,100))
        emb3 = torch.nn.AdaptiveAvgPool1d(10)(y3.view(1,100))
        z1_loss = nn.MSELoss()(z1.float(), emb1.detach().float())
        z2_loss = nn.MSELoss()(z2.float(), emb2.detach().float())
        z3_loss = nn.MSELoss()(z3.float(), emb3.detach().float())-nn.MSELoss()(z2.float(), emb2.detach().float())-nn.MSELoss()(z1.float(), emb1.detach().float())
        optimizer_z1.zero_grad()
        optimizer_z2.zero_grad()
        optimizer_z3.zero_grad()
        z1_loss.backward()
        z2_loss.backward()
        z3_loss.backward()
        optimizer_z1.step()
        optimizer_z2.step()
        optimizer_z3.step()
    linear_model=unlinear_predictor(1,10,1)
    linear_optimizer=torch.optim.Adam(linear_model.parameters(),lr=0.001)

    for i in range(2):

        for cnt, d in enumerate(y2):
            pre2 = linear_model(d, z2)
            try:
                loss = nn.MSELoss()(pre2, y2[cnt + 1])
                linear_optimizer.zero_grad()
            except:
                continue

        for cnt, d in enumerate(y3):
            pre3 = linear_model(d, z3)
            try:
                loss = nn.MSELoss()(pre3, y3[cnt + 1])
                linear_optimizer.zero_grad()
            except:
                continue

        for cnt,d in enumerate(y1):
            pre1=linear_model(d,z1)
            try:
                loss=nn.MSELoss()(pre1,y1[cnt+1])
                linear_optimizer.zero_grad()
            except:
                continue
                
    pre=[]
    for cnt, d in enumerate(y3):
        pre3 = linear_model(d, z3)
        pre.append(float(pre3.detach().numpy()))

    plt.title('sub_optimal1')
    plt.scatter(list(range(len(y1))), y1)
    plt.savefig('sub_optimal1.pdf')
    plt.close()
    plt.title('sub_optimal2')
    plt.scatter(list(range(len(y2))), y2)
    plt.savefig('sub_optimal2.pdf')
    plt.close()
    plt.title('demo')
    plt.scatter(list(range(len(y3))), y3)
    plt.savefig('demo.pdf')
    plt.close()
    plt.title('stitched')
    plt.scatter(list(range(len(pre))), pre)
    plt.savefig('stitched.pdf')


