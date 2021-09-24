import os
import torch
import pickle
from pointnet.model import PointNetVAE
from pointnet.config import *
from model import WGAN_GP
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AE_LOAD_PATH = os.path.join("..", "experiments", model_name, model_params_subdir, epoch_load + ".pth")
NUM_GENERATIONS = 8

# def clip_orientation(d, threshold=0.0):
#     '''
#     clip to [+-1, +-1] if close enough
#     '''
#     major_orientations = [
#         np.array([0, 1]),
#         np.array([0, -1]),
#         np.array([1, 0]),
#         np.array([-1, 0]),
#     ]
#     max_index = -1
#     max_dot = 0
#     for idx, major_orientation in enumerate(major_orientations):
#         dot = np.dot(d, major_orientation)
#         if dot > max_dot:
#             max_dot = dot
#             max_index = idx
#     if max_dot > threshold:
#         return major_orientations[max_index]
#     else:
#         return d

model_ae = PointNetVAE()
model_ae.load_state_dict(torch.load(AE_LOAD_PATH))
model_ae = model_ae.eval()

model_gan = WGAN_GP()
model_gan.load_model("discriminator.pkl", "generator.pkl")
model_gan.eval()

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

base_dir = os.path.join("..", data_dir, room_name)
rooms_dir = os.path.join(base_dir, rooms_subdir)

with open(os.path.join(base_dir, "categories.pkl"), "rb") as f:
    categories_reverse_dict = pickle.load(f)

for _ in range(NUM_GENERATIONS):
    latent_code_sample = model_gan.generate()
    generated_scene = model_ae.generate(latent_code=latent_code_sample)
    print(generated_scene)

    w, h = 500, 500
    scale = 0.25

    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w, h/2)], fill="gray")

    generated_scene = generated_scene.squeeze().detach()
    areas = generated_scene[:, 2] * generated_scene[:, 3]
    indices_area_descending = np.argsort(-areas)
    generated_scene = generated_scene[indices_area_descending]
    for idx, r in enumerate(generated_scene.squeeze().tolist()):
        pos = r[0:2]
        dim = r[2:4]
        ori = r[4:6]
        ori = clip_orientation(ori / np.linalg.norm(ori))
        cat = categories_reverse_dict[np.argmax(r[geometry_size+orientation_size:geometry_size+orientation_size+num_categories])]
        existence = r[-1] > 0

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