import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as func

class GenSAGAN(nn.Module):
    def __init__(self, image_size=32, z_dim=32, conv_dim=64):
        super(GenSAGAN, self).__init__()
        repeat_num = int(np.log2(image_size)) - 3
        mult = 2 ** repeat_num
        self.layer1 = nn.ConvTranspose2d(z_dim, conv_dim*mult, 4)
        self.bn1 = nn.BatchNorm2d(conv_dim*mult)
        self.layer2 = nn.ConvTranspose2d(conv_dim*mult, (conv_dim*mult)//2, 3, 2, 2)
        self.bn2 = nn.BatchNorm2d((conv_dim*mult)//2)
        self.layer3 = nn.ConvTranspose2d((conv_dim*mult)//2, (conv_dim*mult)//4, 3, 2, 2)
        self.bn3 = nn.BatchNorm2d((conv_dim*mult)//4)
        self.layer4 = nn.ConvTranspose2d(64, 1, 2, 2, 1)
        self.attn1 = SAttn(64)
        self.attn2 = SAttn(64)
        self.conv1d = nn.ConvTranspose1d(144, 128, 1)
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        out = func.relu(self.layer1(x))
        out = self.bn1(out)
        out = func.relu(self.layer2(out))
        out = self.bn2(out)
        out = func.relu(self.layer3(out))
        out = self.bn3(out)        
        out ,  p1 = self.attn1(out)
        out = self.layer4(out)
        out = out.view(-1, 1, 144)
        out = out.transpose(1, 2)
        out = self.conv1d(out)
        out = out.transpose(2, 1)
        out = out.view(-1, 128)
        return out , p1


class DiscSAGAN(nn.Module):
    def __init__(self, image_size=32, conv_dim=64):
        super(DiscSAGAN, self).__init__()
        self.layer1 = nn.Conv2d(1, conv_dim, 3, 2, 2)
        self.layer2 = nn.Conv2d(conv_dim, conv_dim*2, 3, 2, 2)
        self.layer3 = nn.Conv2d(conv_dim*2, conv_dim*4, 3 ,2, 2)
        self.layer4 = nn.Conv2d(conv_dim*4, 1, 4)
        self.attn1 = SAttn(256)
        self.attn2 = SAttn(512)
        self.conv1d = nn.ConvTranspose1d(128, 144, 1)
    def forward(self, x):
        # x = x.squeeze(1)
        x = x.unsqueeze(-1)
        x = self.conv1d(x)
        x = x.transpose(2, 1)
        x = x.view(-1, 1, 12, 12)
        out = func.leaky_relu(self.layer1(x))
        out = func.leaky_relu(self.layer2(out))
        out = func.leaky_relu(self.layer3(out))
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out = out.reshape(x.shape[0], -1)
        return out, p1

class SAttn(nn.Module):
    def __init__(self, dim):
        super(SAttn, self).__init__()
        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim//8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        batch_size, c, w, h = x.size()
        query = self.query(x)
        query = query.view(batch_size, -1, w*h).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, w*h)        
        matmul = torch.bmm(query, key)
        attn = self.softmax(matmul)
        value = self.value(x).view(batch_size, -1, w*h)
        out = torch.bmm(value, attn.permute(0,2,1))
        out = out.view(batch_size, c, w, h)
        out = self.gamma*out + x
        return out, attn