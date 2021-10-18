import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from pointnetae.config import *


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
        self.conv5 = torch.nn.Conv1d(128, latent_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(latent_size)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

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
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, latent_size)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class ShapeCodeDecoder(nn.Module):
    def __init__(self):
        super(ShapeCodeDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_size + geometry_size + orientation_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, shape_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PointNetAE(nn.Module):
    def __init__(self, feature_transform=False):
        super(PointNetAE, self).__init__()
        self.feature_transform = feature_transform
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(latent_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, (point_size_intermediate + 1) * max_num_points)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(p=0.3)

        # use a decoder for each shape category
        self.shape_decoders = nn.ModuleList([ShapeCodeDecoder() for _ in range(num_categories)])

    def forward(self, x): # x: [batch_size, point_size + 1, num_points]
        latent_code, trans, trans_feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(latent_code)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1, max_num_points, point_size_intermediate + 1)

        return x, latent_code, trans, trans_feat
    
    def decode_shape(self, x, category_idx): # x: [batch_size=1, latent_size + geometry_size + orientation_size], category_idx: idx in range(0, num_categories)
        shape_decoder = self.shape_decoders[category_idx]
        return shape_decoder(x)
    
    def generate(self, latent_code=None): # note this does not make use of encoder
        if latent_code is None:
            x = torch.randn(1, latent_size)
        else:
            x = latent_code
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        if latent_code.shape[0] == 1:
            x = x.view(max_num_points, point_size + 1)
        else:
            x = x.view(-1, max_num_points, point_size + 1)
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


# def feature_transform_regularizer(trans):
#     d = trans.size()[1]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.cuda()
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
#     return loss


# if __name__ == '__main__':
#     sim_data = Variable(torch.rand(32,point_size + 1,2500))
#     # trans = STN3d()
#     # out = trans(sim_data)
#     # print('stn', out.size())
#     # print('loss', feature_transform_regularizer(out))

#     sim_data_64d = Variable(torch.rand(32, 64, 2500))
#     trans = STNkd(k=64)
#     out = trans(sim_data_64d)
#     print('stn64d', out.size())
#     print('loss', feature_transform_regularizer(out))

#     point_encoder = PointNetEncoder()
#     out, _, _, _, _ = point_encoder(sim_data)
#     print('global feat', out.size())

#     point_vae = PointNetAE()
#     out, _, _, _, _ = point_vae(sim_data)
#     print('VAE', out.size())

#     # pointfeat = PointNetEncoder(global_feat=False)
#     # out, _, _ = pointfeat(sim_data)
#     # print('point feat', out.size())

#     # cls = PointNetCls(k = 5)
#     # out, _, _ = cls(sim_data)
#     # print('class', out.size())

#     # seg = PointNetDenseCls(k = 3)
#     # out, _, _ = seg(sim_data)
#     # print('seg', out.size())
