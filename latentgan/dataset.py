import os
import torch
import numpy as np

from latentgan.config import *


class SceneLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        max_num_points,
        encoder,
        load_ram=False
    ):
        self.data_source = data_source
        self.npyfiles = os.listdir(data_source)
        self.load_ram = load_ram
        self.max_num_points = max_num_points
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filepath = os.path.join(self.data_source, f)
                self.loaded_data.append(self.get_item_from_filepath(filepath))

    def get_item_from_filepath(self, filepath):
        furniture_arr = np.load(filepath)
        num_points = furniture_arr.shape[0]
        assert num_points <= self.max_num_points

        furniture_tensor = np.zeros((num_points, point_size + 1))
        furniture_tensor[0:num_points, 0:geometry_size + orientation_size] = furniture_arr[:, 0:geometry_size + orientation_size] # geometry, orientation
        furniture_tensor[np.arange(num_points), geometry_size + orientation_size + furniture_arr[:, geometry_size + orientation_size].astype(int)] = 1 # category
        furniture_tensor[0:num_points, geometry_size + orientation_size + num_categories] = 1 # existence (TODO: remove this from everywhere)
        furniture_tensor[0:num_points, geometry_size + orientation_size + num_categories + 1:] = furniture_arr[:, geometry_size + orientation_size + 1:] # shape

        scene = torch.Tensor(furniture_tensor).transpose(1, 0)
        cats = furniture_arr[:, geometry_size + orientation_size].astype(int) # category indices

        return self.encoder(scene.unsqueeze(0), np.expand_dims(cats, 0))[0]

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        if self.load_ram:
            return self.loaded_data[idx]
        else:
            filepath = os.path.join(self.data_source, self.npyfiles[idx])
            return self.get_item_from_filepath(filepath)
    
    def get_room_id(self, idx):
        return os.path.splitext(self.npyfiles[idx])[0]