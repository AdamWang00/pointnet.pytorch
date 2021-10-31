import os
import torch
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from pointnetae.model import PointNetAE
from pointnetae.config import *
from pointnetae.utils import *
from pointnetae.dataset import SceneDataset

LOAD_PATH = os.path.join("experiments", model_name, model_params_subdir, epoch_load + ".pth")
NUM_TESTS = 8
DATASET_OFFSET = 0
HIDE_NONEXISTENT_OUTPUTS = True

model = PointNetAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval().cuda()

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

base_dir = os.path.join(data_dir, room_name)
rooms_dir = os.path.join(base_dir, rooms_subdir)

with open(os.path.join(base_dir, "categories.json"), "r") as f:
    categories_reverse_dict = json.load(f)

scene_dataset = SceneDataset(rooms_dir, max_num_points)

for i in range(DATASET_OFFSET, DATASET_OFFSET + NUM_TESTS):
    scene, target = scene_dataset.__getitem__(i)
    scene = scene.transpose(1, 0).cuda()
    cats = target[:, geometry_size + orientation_size].numpy().astype(int)

    reconstruction, latent_code = model(scene.unsqueeze(0), np.expand_dims(cats, 0))
    reconstruction = reconstruction[0].detach().cpu()

    cost_mat_position = get_cost_matrix_2d(reconstruction[:, 0:2], target[:, 0:2])
    cost_mat_dimension = get_cost_matrix_2d(reconstruction[:, 2:4], target[:, 2:4])
    cost_mat = cost_mat_position + dimensions_matching_weight * cost_mat_dimension
    cost_mat = cost_mat.detach()
    target_ind, matched_ind, unmatched_ind = get_assignment_problem_matchings(cost_mat)

    reconstruction_matched = reconstruction[matched_ind]
    reconstruction_unmatched = reconstruction[unmatched_ind]
    target_existence = torch.zeros(max_num_points)
    target_existence[matched_ind] = 1
    target = target[target_ind]

    # print("===== WEIGHTED LOSSES =====")

    # print("geometry loss:", geometric_weight * geometric_loss(
    #     reconstruction_matched[:, 0:geometry_size],
    #     target[:, 0:geometry_size]
    # ).item())

    # print("orientation loss:", orientation_weight * orientation_loss(
    #     reconstruction_matched[:, geometry_size:geometry_size+orientation_size],
    #     target[:, geometry_size:geometry_size+orientation_size]
    # ).item())

    # print("categorical loss:", categorical_weight * categorical_loss(
    #     reconstruction_matched[:, geometry_size:geometry_size+orientation_size+num_categories],
    #     target[:, geometry_size+orientation_size].long()
    # ).item())

    # print("existence loss:", existence_weight * existence_loss(
    #     reconstruction[:, geometry_size+orientation_size+num_categories],
    #     target_existence
    # ).item())

    # for matched_index in range(target.shape[0]):
    #     print(f'===== MATCH {matched_index + 1} =====')
    #     print("Geometry Predicted:", reconstruction_matched[matched_index, 0:geometry_size].tolist())
    #     print("Geometry Actual:", target[matched_index, 0:geometry_size].tolist())
    #     print("Orientation Predicted:", reconstruction_matched[matched_index, geometry_size:geometry_size+orientation_size].tolist())
    #     print("Orientation Actual:", target[matched_index, geometry_size:geometry_size+orientation_size].tolist())
    #     print("Category Predicted:", reconstruction_matched[matched_index, geometry_size+orientation_size:geometry_size+orientation_size+num_categories].argmax().item())
    #     print("Category Actual:", target[matched_index, geometry_size+orientation_size].long().item())
    #     print("Existence Predicted:", reconstruction_matched[matched_index, geometry_size+orientation_size+num_categories].item() > 0)

    # for unmatched_index in range(reconstruction_unmatched.shape[0]):
    #     print(f'===== NON-MATCH {unmatched_index + 1} =====')
    #     print("Geometry Predicted:", reconstruction_unmatched[unmatched_index, 0:geometry_size].tolist())
    #     print("Orientation Predicted:", reconstruction_unmatched[unmatched_index, geometry_size:geometry_size+orientation_size].tolist())
    #     print("Category Predicted:", reconstruction_unmatched[unmatched_index, geometry_size+orientation_size:geometry_size+orientation_size+num_categories].argmax().item())
    #     print("Existence Predicted:", reconstruction_unmatched[unmatched_index, geometry_size+orientation_size+num_categories].item() > 0)

    w, h = 500, 500
    scale = 0.2

    img = Image.new("RGB", (2*w, h), color="white")
    draw = ImageDraw.Draw(img)  

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(w + w/2, 0), (w + w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w + w, h/2)], fill="gray")
    draw.rectangle([(w, 0), (w, h)], fill="black")

    areas = target[:, 2] * target[:, 3]
    indices_area_descending = np.argsort(-areas)
    target = target[indices_area_descending]
    for t in target.tolist():
        pos = t[0:2]
        dim = t[2:4]
        ori = t[4:6]
        ori = clip_orientation(ori / np.linalg.norm(ori))
        cat = categories_reverse_dict[str(int(t[6]))]

        if (ori[1] == 0): # Need to flip dimensions if oriented towards East/West
            dim[0], dim[1] = dim[1], dim[0]

        box_nw = (w/2*(1 + (pos[0] - dim[0]/2)*scale), h/2*(1 + (pos[1] - dim[1]/2)*scale))
        box_se = (w/2*(1 + (pos[0] + dim[0]/2)*scale), h/2*(1 + (pos[1] + dim[1]/2)*scale))

        draw.rectangle(
            [box_nw, box_se],
            fill=(colors[cat]),
            width=3,
            outline="black"
        )

        box_center = ((box_nw[0] + box_se[0])/2, (box_nw[1] + box_se[1])/2)
        ori_indicator = (box_center[0] + ori[0] * w * dim[0] / 4 * scale, box_center[1] + ori[1] * h * dim[1] / 4 * scale)
        draw.line([box_center, ori_indicator], fill='white')

    reconstruction = reconstruction.squeeze().detach()
    areas = reconstruction[:, 2] * reconstruction[:, 3]
    indices_area_descending = np.argsort(-areas)
    reconstruction = reconstruction[indices_area_descending]
    for idx, r in zip(indices_area_descending.tolist(), reconstruction.tolist()):
        pos = r[0:2]
        dim = r[2:4]
        ori = r[4:6]
        ori = clip_orientation(ori / np.linalg.norm(ori))
        cat = categories_reverse_dict[str(np.argmax(r[geometry_size+orientation_size:geometry_size+orientation_size+num_categories]))]
        existence = r[geometry_size+orientation_size+num_categories] > 0

        if HIDE_NONEXISTENT_OUTPUTS and not existence:
            continue

        if (ori[1] == 0): # Need to flip dimensions if oriented towards East/West
            dim[0], dim[1] = dim[1], dim[0]

        box_nw = (w/2*(3 + (pos[0] - dim[0]/2)*scale), h/2*(1 + (pos[1] - dim[1]/2)*scale))
        box_se = (w/2*(3 + (pos[0] + dim[0]/2)*scale), h/2*(1 + (pos[1] + dim[1]/2)*scale))

        if existence:
            draw.rectangle(
                [box_nw, box_se],
                fill=colors[cat],
                width=3,
                outline="black"
            )
        else:
            draw.rectangle(
                [box_nw, box_se],
                fill=colors_light[cat],
                width=1,
                outline="gray"
            )
        
        box_center = ((box_nw[0] + box_se[0])/2, (box_nw[1] + box_se[1])/2)
        ori_indicator = (box_center[0] + ori[0] * w * dim[0] / 4 * scale, box_center[1] + ori[1] * h * dim[1] / 4 * scale)
        draw.line([box_center, ori_indicator], fill='white')

        box_label_nw = (box_nw[0] + 2, box_nw[1] + 2)
        draw.text(box_label_nw, f'{idx}', fill=(200,200,200), font=font)

    img.show()