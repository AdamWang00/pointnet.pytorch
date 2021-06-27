from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .config import *


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        # self.stn = STNkd(k=3)

        self.conv1 = torch.nn.Conv1d(point_size + 1, 64, 1) # temporarily use (point_size + 1) instead of (point_size) to include existence label, in order to zero-pad the input up to some max num_points
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.fc_mu = nn.Linear(1024, 1024)
        self.fc_var = nn.Linear(1024, 1024)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x): # x: [batch_size, point_size + 1, num_points]
        n_pts = x.size()[2]

        trans = None
        # trans = self.stn(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        print("TODO: check if we need to use x.view", x.shape)
        x = x.view(-1, 1024)

        # VAE (no activation)
        mu = self.conv5_mu(x)
        log_var = self.conv5_var(x)
        x = self.reparameterize(mu, log_var)

        if self.global_feat:
            return x, trans, trans_feat, mu, log_var
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, mu, log_var


class PointNetVAE(nn.Module):
    def __init__(self, feature_transform=False):
        super(PointNetVAE, self).__init__()
        self.feature_transform = feature_transform
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, (point_size + 1) * max_num_points)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(p=0.3)

    def forward(self, x): # x: [batch_size, point_size + 1, num_points]
        x, trans, trans_feat, mu, log_var = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1, max_num_points, point_size + 1)

        return x, trans, trans_feat, mu, log_var
    
    def generate(self):
        x = torch.randn(1, 1024)
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        x = x.view(max_num_points, point_size + 1)

        return x


# class PointNetCls(nn.Module):
#     def __init__(self, k=2, feature_transform=False):
#         super(PointNetCls, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1), trans, trans_feat


# class PointNetDenseCls(nn.Module):
#     def __init__(self, k = 2, feature_transform=False):
#         super(PointNetDenseCls, self).__init__()
#         self.k = k
#         self.feature_transform=feature_transform
#         self.feat = PointNetEncoder(global_feat=False, feature_transform=feature_transform)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)

#     def forward(self, x):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2,1).contiguous()
#         x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batchsize, n_pts, self.k)
#         return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,point_size + 1,2500))
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    point_encoder = PointNetEncoder()
    out, _, _, _, _ = point_encoder(sim_data)
    print('global feat', out.size())

    point_vae = PointNetVAE()
    out, _, _, _, _ = point_vae(sim_data)
    print('VAE', out.size())

    # pointfeat = PointNetEncoder(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())

    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())
