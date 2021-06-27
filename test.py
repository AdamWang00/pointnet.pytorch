import torch
from pointnet.model import PointNetVAE
from pointnet.config import *
from utils import *

LOAD_PATH = "experiments/model1/9.pth"
model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

scene, target_list = generate_scene(1)
scene = scene.transpose(2, 1)
target = target_list[0]

reconstruction, _, _, mu, log_var = model(scene)
reconstruction = reconstruction[0]

cost_mat = get_cost_matrix_2d(reconstruction[:, 0:2], target[:, 0:2])
cost_mat = cost_mat.detach()
target_ind, matched_ind, unmatched_ind = get_assignment_problem_matchings(cost_mat)

reconstruction_matched = reconstruction[matched_ind]
reconstruction_unmatched = reconstruction[unmatched_ind]
target_existence = torch.zeros(max_num_points)
target_existence[matched_ind] = 1
target = target[target_ind]

print("geometry loss:", geometry_loss(
    reconstruction_matched[:, 0:geometry_size],
    target[:, 0:geometry_size]
).item())

print("categorical loss:", categorical_loss(
    reconstruction_matched[:, geometry_size:geometry_size+num_classes],
    target[:, geometry_size].long()
).item())

print("existence loss:", existence_loss(
    reconstruction[:, geometry_size+num_classes],
    target_existence
).item())

mu = mu.view(-1, mu.shape[1] * mu.shape[2])
log_var = log_var.view(-1, log_var.shape[1] * log_var.shape[2])
print("weighted kld loss:", kld_loss_weight * kld_loss(mu, log_var).item())

for matched_index in range(target.shape[0]):
    print(f'===== MATCH {matched_index + 1} =====')
    print("Geometry Predicted:", reconstruction_matched[matched_index, 0:geometry_size].tolist())
    print("Geometry Actual:", target[matched_index, 0:geometry_size].tolist())
    print("Category Predicted:", reconstruction_matched[matched_index, geometry_size:geometry_size+num_classes].argmax().item())
    print("Category Actual:", target[matched_index, geometry_size].long().item())
    print("Existence Predicted:", torch.sigmoid(reconstruction_matched[matched_index, geometry_size+num_classes]).item() > 0.5)

for unmatched_index in range(reconstruction_unmatched.shape[0]):
    print(f'===== NON-MATCH {unmatched_index + 1} =====')
    print("Geometry Predicted:", reconstruction_unmatched[unmatched_index, 0:geometry_size].tolist())
    print("Category Predicted:", reconstruction_unmatched[unmatched_index, geometry_size:geometry_size+num_classes].argmax().item())
    print("Existence Predicted:", torch.sigmoid(reconstruction_unmatched[unmatched_index, geometry_size+num_classes]).item() > 0.5)