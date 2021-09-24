import numpy as np
import torch
import torch.nn.functional as F
import random
from numpy.random import normal
from scipy.optimize import linear_sum_assignment
from torch.nn.modules.module import T
from pointnetvae.config import *


def geometric_loss(x, target):
    """
    x: (N, geometry_size)
    target: (N, geometry_size)
    """
    # print("GEOMETRY", x, target)
    assert len(x.shape) == 2
    assert x.shape == target.shape
    return F.mse_loss(x, target)


def orientation_loss(x, target):
    """
    x: (N, orientation_size)
    target: (N, orientation_size)
    """
    assert len(x.shape) == 2
    assert x.shape == target.shape
    return 1 + torch.mean(-F.cosine_similarity(x, target))


def categorical_loss(x_logits, target_class):
    """
    x_logits: (N, num_categories)
    target_class: (N) where each value l (label) is 0 <= l <= num_categories-1
    """
    # print("CATEGORICAL", x_logits, target_class)
    assert len(x_logits.shape) == 2
    assert len(target_class.shape) == 1
    assert x_logits.shape[0] == target_class.shape[0]
    return F.cross_entropy(x_logits, target_class)


def existence_loss(x_logits, target_existence):
    """
    x_logits: (N)
    target_existence: (N) where each value is 0 or 1
    """
    # print("EXISTENCE", x_logits, target_existence)
    assert len(x_logits.shape) == 1
    assert x_logits.shape == target_existence.shape
    return F.binary_cross_entropy_with_logits(x_logits, target_existence)


def kld_loss(mu, log_var):
    """
    mu: (N, latent_size)
    log_var: (N, latent_size)
    """
    assert len(mu.shape) == 2
    assert mu.shape == log_var.shape
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


def get_assignment_problem_matchings(cost_matrix):
    """
    cost_matrix must have r rows and c cols (shape (r, c)), where r <= c

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
    x_pos: (max_num_points, D)
    target_pos: (num_points, D), where num_points <= max_num_points
    returns (num_points, max_num_points)
    """
    assert x_pos.shape[0] >= target_pos.shape[0]
    assert x_pos.shape[1] == target_pos.shape[1]
    x_pos_repeated = x_pos.repeat(target_pos.shape[0], 1, 1)
    target_pos_repeated = target_pos.repeat(x_pos.shape[0], 1, 1).transpose(0, 1)
    cost_matrix = torch.norm(x_pos_repeated - target_pos_repeated, 2, -1)
    return cost_matrix


def generate_scene(batch_size, encoding=None):
    if procedure == "table1":
        return table1(batch_size, encoding=encoding)
    elif procedure == "table2":
        return table2(batch_size)
    else:
        return None


def table1(batch_size, encoding=None):
    """
    "a square table with 0-4 chairs"
    [max_num_points, point_size + 1], [max_num_points, geometry_size + orientation_size + 1 + code_size]
    (padded with zeros)

    encoding: None for random, 0-15 for nonrandom
    """
    assert max_num_points >= 5

    scene = np.zeros((batch_size, max_num_points, point_size + 1))
    target_list = []

    for i in range(batch_size):
        # "table"
        scene[i, 0, :] = [0, 0, 0.4, 0.4, 1, 0, 1]
        target = [[0, 0, 0.4, 0.4, 0]]

        # "chairs"
        if encoding is None:
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
        else:
            if encoding % 2 == 1:
                scene[i, 1, :] = [0.4, 0, 0.2, 0.2, 0, 1, 1]
                target.append([0.4, 0, 0.2, 0.2, 1])
            if (encoding // 2) % 2 == 1:
                scene[i, 2, :] = [0, 0.4, 0.2, 0.2, 0, 1, 1]
                target.append([0, 0.4, 0.2, 0.2, 1])
            if (encoding // 4) % 2 == 1:
                scene[i, 3, :] = [-0.4, 0, 0.2, 0.2, 0, 1, 1]
                target.append([-0.4, 0, 0.2, 0.2, 1])
            if (encoding // 8) % 2 == 1:
                scene[i, 4, :] = [0, -0.4, 0.2, 0.2, 0, 1, 1]
                target.append([0, -0.4, 0.2, 0.2, 1])
            target_list.append(torch.Tensor(target))
    
    return (torch.Tensor(scene), target_list)


def table2(batch_size):
    """
    "a square table with 4 chairs of varying distance (gaussian) from the table"
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
        rand_dist = np.clip(normal(0.5, 0.1), 0.3, 0.7)
        scene[i, 1, :] = [rand_dist, 0, 0.2, 0.2, 0, 1, 1]
        target.append([rand_dist, 0, 0.2, 0.2, 1])

        rand_dist = np.clip(normal(0.5, 0.1), 0.3, 0.7)
        scene[i, 2, :] = [0, rand_dist, 0.2, 0.2, 0, 1, 1]
        target.append([0, rand_dist, 0.2, 0.2, 1])

        rand_dist = np.clip(normal(0.5, 0.1), 0.3, 0.7)
        scene[i, 3, :] = [-rand_dist, 0, 0.2, 0.2, 0, 1, 1]
        target.append([-rand_dist, 0, 0.2, 0.2, 1])

        rand_dist = np.clip(normal(0.5, 0.1), 0.3, 0.7)
        scene[i, 4, :] = [0, -rand_dist, 0.2, 0.2, 0, 1, 1]
        target.append([0, -rand_dist, 0.2, 0.2, 1])

        target_list.append(torch.Tensor(target))
    
    return (torch.Tensor(scene), target_list)

def clip_orientation(d, threshold=0.0):
    '''
    clip to [+-1, +-1] if close enough
    '''
    major_orientations = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ]
    max_index = -1
    max_dot = 0
    for idx, major_orientation in enumerate(major_orientations):
        dot = np.dot(d, major_orientation)
        if dot > max_dot:
            max_dot = dot
            max_index = idx
    if max_dot > threshold:
        return major_orientations[max_index]
    else:
        return d