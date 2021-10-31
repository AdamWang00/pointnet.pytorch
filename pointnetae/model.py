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


class ShapeCodeEncoder(nn.Module):
    def __init__(self, output_size):
        super(ShapeCodeEncoder, self).__init__()
        dims = [shape_size] + shape_code_encoder_hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], output_size))
        self.main_module = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.main_module(x)


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()

        # To add batch norm, add after FC but before relu

        self.shape_code_encoder_output_size = shape_code_encoder_output_size
        self.intermediate_size = point_size + 1 - shape_size + self.shape_code_encoder_output_size # temporarily use (point_size + 1) instead of (point_size) to include existence label, in order to zero-pad the input up to some max num_points

        dims = [self.intermediate_size] + encoder_hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Conv1d(dims[i], dims[i + 1], 1))
            modules.append(nn.ReLU(True))
        modules.append(nn.Conv1d(dims[-1], latent_size, 1))
        self.main_module = nn.Sequential(*modules)
        
        # use an encoder for each shape category
        self.shape_encoders = nn.ModuleList([ShapeCodeEncoder(self.shape_code_encoder_output_size) for _ in range(num_categories)])

    def forward(self, x, cats): # x: [batch_size, point_size + 1, num_points], cats: [batch_size, num_points]
        b_size, _, n_pts = x.size()

        # to add batching (i.e. to keep x_intermediate consistently shaped), replace n_pts with max_num_points
        x_intermediate = torch.zeros((b_size, self.intermediate_size, n_pts)).cuda()
        x_intermediate[:, 0:point_size + 1 - shape_size, :] = x[:, 0:point_size + 1 - shape_size, :]
        for b in range(b_size): # for each batch (i.e. room)
            for p in range(n_pts): # for each point in the batch (i.e. furniture)
                shape_encoder = self.shape_encoders[cats[b, p]]
                x_intermediate[b, -self.shape_code_encoder_output_size:, p] = shape_encoder(x[b, -shape_size:, p])
        x = x_intermediate

        x = self.main_module(x)
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, latent_size)

        return x


class ShapeCodeDecoder(nn.Module):
    def __init__(self):
        super(ShapeCodeDecoder, self).__init__()
        dims = [latent_size + geometry_size + orientation_size] + shape_code_decoder_hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], shape_size))
        self.main_module = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.main_module(x)


class PointNetAE(nn.Module):
    def __init__(self, feature_transform=False):
        super(PointNetAE, self).__init__()
        self.feature_transform = feature_transform
        self.encoder = PointNetEncoder()

        # Decoder:
        # To add batch norm, add after FC but before relu
        dims = [latent_size] + decoder_hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], (point_size_intermediate + 1) * max_num_points))
        self.main_module = nn.Sequential(*modules)

        # use a shape code decoder for each shape category
        self.shape_decoders = nn.ModuleList([ShapeCodeDecoder() for _ in range(num_categories)])

    def forward(self, x, cats): # x: [batch_size, point_size + 1, num_points], cats: [batch_size, num_points]
        latent_code = self.encoder(x, cats)

        x = self.main_module(latent_code)
        x = x.view(-1, max_num_points, point_size_intermediate + 1)

        return x, latent_code

    def decode_shape(self, x, category_idx): # x: [batch_size=1, latent_size + geometry_size + orientation_size], category_idx: idx in range(0, num_categories)
        shape_decoder = self.shape_decoders[category_idx]
        return shape_decoder(x)
    
    def generate(self, latent_code=None): # note this does not make use of encoder
        if latent_code is None:
            x = torch.randn(1, latent_size)
        else:
            x = latent_code

        x = self.main_module(x)

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
