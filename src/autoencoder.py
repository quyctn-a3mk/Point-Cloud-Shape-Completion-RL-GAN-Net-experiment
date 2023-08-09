import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pyntcloud import PyntCloud

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = [nn.Conv1d(3, 64, kernel_size=1), 
                nn.BatchNorm1d(64),
                nn.ReLU()]
        conv2 = [nn.Conv1d(64, 128, kernel_size=1), 
                nn.BatchNorm1d(128),
                nn.ReLU()]
        conv3 = [nn.Conv1d(128, 256, kernel_size=1), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        conv4 = [nn.Conv1d(256, 128, kernel_size=1), 
                nn.BatchNorm1d(128),
                nn.AdaptiveMaxPool1d(1)]
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)        
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)        
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        print(out_4.shape)
        out_4 = out_4.view(-1, out_4.shape[1])
        print(out_4.shape)
        return out_4

class Decoder(nn.Module):
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        linear1 = [nn.Linear(128, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        linear2 = [nn.Linear(256, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        linear3 = [nn.Linear(256, 6144), 
                nn.ReLU()]
        self.linear1 = nn.Sequential(*linear1)
        self.linear2 = nn.Sequential(*linear2)
        self.linear3 = nn.Sequential(*linear3)
        self.num_points = num_points        
    def forward(self, x):
        out_1 = self.linear1(x)
        out_2 = self.linear2(out_1)
        out_3 = self.linear3(out_2)        
        return out_3.view(-1, 3, self.num_points)

class AutoEncoder(nn.Module):
    def __init__(self, num_points):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)
    def forward(self, x):
        gfv = self.encoder(x)
        out = self.decoder(gfv)
        return out, gfv   
    def encode(self, x):
        gfv = self.encoder(x)
        # out = self.decoder(gfv)
        return gfv
    def decode(self, x):
        return self.decoder(x)

class PointcloudDatasetAE(Dataset):
    def __init__(self, root, list_point_clouds):
        self.root = root
        self.list_files = list_point_clouds
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self, index):
        points = PyntCloud.from_file(self.list_files[index])
        points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(np.float)
        points = torch.from_numpy(points)
        return points

class PointcloudDatasetNoisy(Dataset):
    def __init__(self, root, list_point_clouds):
        self.root = root
        self.list_files = list_point_clouds        
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self, index):
        points = self.list_files[index]
        # try:
        #     # if points.isdigit():
        #     points = float(points)
        # except ValueError:
        #     return None
        # points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(np.float)
        points = torch.from_numpy(points)        
        return points