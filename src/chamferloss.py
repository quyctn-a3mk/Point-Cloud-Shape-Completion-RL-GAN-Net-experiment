import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChamferLoss_distance(nn.Module):
    def __init__(self, num_points, device = DEVICE):
        super(ChamferLoss_distance, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)
    ##    
    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) - predict_pc.unsqueeze(-1),dim=1), dim=-2)
            # self.loss += z.sum()
        self.loss = z.sum() / (len(gt_pc)*self.num_points)
        ##
        z_2, _ = torch.min(torch.norm(predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1),dim=1), dim=-2)
        self.loss += z_2.sum() / (len(gt_pc)*self.num_points)
        return self.loss

class ChamferLoss_loss(nn.Module):
    def __init__(self, num_points, device = DEVICE):
        super(ChamferLoss_loss, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)
    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) -
                                    predict_pc.unsqueeze(-1), dim=1), dim=-2)
        self.loss = z.sum() / (len(gt_pc)*(gt_pc.shape[2]+predict_pc.shape[2]))
        z_2, _ = torch.min(torch.norm(
            predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1), dim=1), dim=-2)
        self.loss += z_2.sum() / (len(gt_pc)*(gt_pc.shape[2]+predict_pc.shape[2]))
        return self.loss     