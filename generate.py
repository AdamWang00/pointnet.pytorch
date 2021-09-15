import os
import torch
import pickle
from pointnet.model import PointNetVAE
from pointnet.config import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOAD_PATH = os.path.join("experiments", model_name, model_params_subdir, epoch_load + ".pth")
NUM_GENERATIONS = 8

model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

colors = {
    "Bed": "blue",
    "Nightstand": "orange"
}

colors_light = {
    "Bed": "#dbf0fe",
    "Nightstand": "#ffeeda"
}

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

base_dir = os.path.join(data_dir, room_type)
rooms_dir = os.path.join(base_dir, rooms_subdir)

with open(os.path.join(base_dir, "categories.pkl"), "rb") as f:
    categories_reverse_dict = pickle.load(f)

for _ in range(NUM_GENERATIONS):
    generated_scene = model.generate()
    print(generated_scene)

    w, h = 500, 500
    scale = 0.25

    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w, h/2)], fill="gray")

    for idx, r in enumerate(generated_scene.squeeze().tolist()):
        pos = r[0:2]
        dim = r[2:4]
        ori = r[4:6]
        ori /= np.linalg.norm(ori)
        cat = categories_reverse_dict[np.argmax(r[geometry_size+orientation_size:geometry_size+orientation_size+num_categories])]
        existence = r[-1] > 0

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
                outline="black"
            )

        box_center = ((box_nw[0] + box_se[0])/2, (box_nw[1] + box_se[1])/2)
        ori_indicator = (box_center[0] + ori[0] * w * dim[0] / 4 * scale, box_center[1] + ori[1] * h * dim[1] / 4 * scale)
        draw.line([box_center, ori_indicator], fill='white')
        
        box_label_nw = (box_nw[0] + 2, box_nw[1] + 2)
        draw.text(box_label_nw, f'{idx}', fill=(255,255,255,128), font=font)

    img.show()