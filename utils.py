import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy.optimize import linear_sum_assignment

from pointnet.config import *


def geometry_loss(x, target):
    """
    x: (N, geometry_size)
    target: (N, geometry_size)
    """
    return F.mse_loss(x, target)


def categorical_loss(x_logits, target_class):
    """
    x_logits: (N, num_classes)
    target_class: (N) where each value l (label) is 0 <= l <= num_classes-1
    """
    return F.cross_entropy(x_logits, target_class)


def existence_loss(x_logits, target_existence):
    """
    x_logits: (N)
    target_existence: (N) where each value is 0 or 1
    """
    return F.binary_cross_entropy_with_logits(x_logits, target_existence)


def kld_loss(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


def get_assignment_problem_matchings(cost_matrix):
    """
    cost_matrix must have r rows and c cols, where r <= c

    finds matching with minimum total cost

    returns (3):
        indices of r rows;
        indices of r corresponding cols;
        indices of c - r cols that are unmatched
    """
    assert cost_matrix.shape[0] <= cost_matrix.shape[1]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    unmatched_col_ind = np.setxor1d(np.arange(cost_matrix.shape[1]), col_ind)
    return row_ind, col_ind, unmatched_col_ind


def get_cost_matrix_2d(x_pos, target_pos):
    """
    x_pos: (max_num_points, 2)
    target_pos: (num_points, 2), where num_points <= max_num_points
    """
    x_pos_repeated = x_pos.repeat(target_pos.shape[0], 1, 1)
    target_pos_repeated = target_pos.repeat(x_pos.shape[0], 1, 1).transpose(0, 1)
    return torch.norm(x_pos_repeated - target_pos_repeated, 2, -1)


def generate_scene(batch_size):
    """
    "a square table with chairs"
    [max_num_points, point_size + 1], [max_num_points, geometry_size + orientation_size + 1 + code_size]
    (padded with zeros)
    """
    assert max_num_points >= 5

    scene = np.zeros((batch_size, max_num_points, point_size + 1))
    target_list = []

    for i in range(batch_size):
        # "table"
        scene[i, 0, :] = [0, 0, 0.4, 0.4, 1, 0, 1]
        target = [[0, 0, 0.4, 0.4, 0]]

        # "chairs"
        if random.random() < 0.5:
            scene[i, 1, :] = [0.4, 0, 0.2, 0.2, 0, 1, 1]
            target.append([0.4, 0, 0.2, 0.2, 1])
        if random.random() < 0.5:
            scene[i, 2, :] = [0, 0.4, 0.2, 0.2, 0, 1, 1]
            target.append([0, 0.4, 0.2, 0.2, 1])
        if random.random() < 0.5:
            scene[i, 3, :] = [-0.4, 0, 0.2, 0.2, 0, 1, 1]
            target.append([-0.4, 0, 0.2, 0.2, 1])
        if random.random() < 0.5:
            scene[i, 4, :] = [0, -0.4, 0.2, 0.2, 0, 1, 1]
            target.append([0, -0.4, 0.2, 0.2, 1])
        target_list.append(torch.Tensor(target))
    
    return (torch.Tensor(scene), target_list)