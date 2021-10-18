import os
import torch
import numpy as np
from pointnetae.config import *

class SceneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        max_num_points,
        load_ram=False
    ):
        self.data_source = data_source
        self.npyfiles = os.listdir(data_source)
        self.load_ram = load_ram
        self.max_num_points = max_num_points

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filepath = os.path.join(self.data_source, f)
                self.loaded_data.append(self.get_item_from_filepath(filepath))

    def get_item_from_filepath(self, filepath):
        furniture_arr = np.load(filepath)
        num_points = furniture_arr.shape[0]
        assert num_points <= self.max_num_points

        target_tensor = furniture_arr
        furniture_tensor = np.zeros((self.max_num_points, point_size + 1))
        furniture_tensor[0:num_points, 0:geometry_size + orientation_size] = furniture_arr[:, 0:geometry_size + orientation_size] # geometry, orientation
        furniture_tensor[np.arange(num_points), geometry_size + orientation_size + furniture_arr[:, geometry_size + orientation_size].astype(int)] = 1 # category
        furniture_tensor[0:num_points, geometry_size + orientation_size + num_categories] = 1 # existence
        furniture_tensor[0:num_points, geometry_size + orientation_size + num_categories + 1:] = furniture_arr[:, geometry_size + orientation_size + 1:] # shape

        return torch.Tensor(furniture_tensor), torch.Tensor(target_tensor)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        if self.load_ram:
            return self.loaded_data[idx]
        else:
            filepath = os.path.join(self.data_source, self.npyfiles[idx])
            return self.get_item_from_filepath(filepath)