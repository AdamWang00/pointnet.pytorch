import torch
import numpy as np
from pointnet.model import PointNetVAE
from pointnet.config import *
from utils import *
from PIL import Image, ImageDraw, ImageFont

LOAD_PATH = "experiments/" + model_name + "/" + epoch_load + ".pth"
NUM_TESTS = 8

model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

for enc in range(NUM_TESTS):
    scene, target_list = generate_scene(1, encoding=enc)
    target = target_list[0]

    reconstruction, _, _, mu, log_var = model(scene.transpose(2, 1))
    reconstruction = reconstruction[0]

    cost_mat = get_cost_matrix_2d(reconstruction[:, 0:2], target[:, 0:2])
    cost_mat = cost_mat.detach()
    target_ind, matched_ind, unmatched_ind = get_assignment_problem_matchings(cost_mat)

    reconstruction_matched = reconstruction[matched_ind]
    reconstruction_unmatched = reconstruction[unmatched_ind]
    target_existence = torch.zeros(max_num_points)
    target_existence[matched_ind] = 1
    target = target[target_ind]

    print("===== WEIGHTED LOSSES =====")

    print("geometry loss:", geometric_weight * geometric_loss(
        reconstruction_matched[:, 0:geometry_size],
        target[:, 0:geometry_size]
    ).item())

    print("categorical loss:", categorical_weight * categorical_loss(
        reconstruction_matched[:, geometry_size:geometry_size+num_categories],
        target[:, geometry_size].long()
    ).item())

    print("existence loss:", existence_weight * existence_loss(
        reconstruction[:, geometry_size+num_categories],
        target_existence
    ).item())

    print("kld loss:", kld_loss_weight * kld_loss(
        mu,
        log_var
    ).item())

    for matched_index in range(target.shape[0]):
        print(f'===== MATCH {matched_index + 1} =====')
        print("Geometry Predicted:", reconstruction_matched[matched_index, 0:geometry_size].tolist())
        print("Geometry Actual:", target[matched_index, 0:geometry_size].tolist())
        print("Category Predicted:", reconstruction_matched[matched_index, geometry_size:geometry_size+num_categories].argmax().item())
        print("Category Actual:", target[matched_index, geometry_size].long().item())
        print("Existence Predicted:", reconstruction_matched[matched_index, geometry_size+num_categories].item() > 0)

    for unmatched_index in range(reconstruction_unmatched.shape[0]):
        print(f'===== NON-MATCH {unmatched_index + 1} =====')
        print("Geometry Predicted:", reconstruction_unmatched[unmatched_index, 0:geometry_size].tolist())
        print("Category Predicted:", reconstruction_unmatched[unmatched_index, geometry_size:geometry_size+num_categories].argmax().item())
        print("Existence Predicted:", reconstruction_unmatched[unmatched_index, geometry_size+num_categories].item() > 0)

    w, h = 500, 500

    img = Image.new("RGB", (2*w, h), color="white")
    draw = ImageDraw.Draw(img)  

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(w + w/2, 0), (w + w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w + w, h/2)], fill="gray")
    draw.rectangle([(w, 0), (w, h)], fill="black")

    for t in target.tolist():
        draw.rectangle(
            [
                (w/2*(1 + t[0] - t[2]/2), h/2*(1 + t[1] - t[3]/2)),
                (w/2*(1 + t[0] + t[2]/2), h/2*(1 + t[1] + t[3]/2))
            ],
            fill=("blue" if t[4] == 0 else "orange"),
            width=3,
            outline="black"
        )

    for idx, r in enumerate(reconstruction.squeeze().tolist()):
        category = np.argmax(r[geometry_size:geometry_size+num_categories])
        existence = r[geometry_size+num_categories] > 0

        box_nw = (w/2*(3 + r[0] - r[2]/2), h/2*(1 + r[1] - r[3]/2))
        box_se = (w/2*(3 + r[0] + r[2]/2), h/2*(1 + r[1] + r[3]/2))

        if existence:
            draw.rectangle(
                [box_nw, box_se],
                fill=("blue" if category == 0 else "orange"),
                width=3,
                outline="black"
            )
        else:
            draw.rectangle(
                [box_nw, box_se],
                fill=("#dbf0fe" if category == 0 else "#ffeeda"),
                width=1,
                outline="black"
            )

        box_label_nw = (box_nw[0] + 2, box_nw[1] + 2)
        draw.text(box_label_nw, f'{idx}', fill=(255,255,255,128), font=font)

    img.show()