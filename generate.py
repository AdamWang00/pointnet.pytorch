import os
import torch
from pointnet.model import PointNetVAE
from pointnet.config import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOAD_PATH = os.path.join("experiments", model_name, model_params_subdir, epoch_load + ".pth")
NUM_GENERATIONS = 8

model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

for _ in range(NUM_GENERATIONS):
    generated_scene = model.generate()

    w, h = 500, 500

    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w, h/2)], fill="gray")

    for idx, r in enumerate(generated_scene.squeeze().tolist()):
        category = np.argmax(r[geometry_size:geometry_size+num_categories])
        existence = r[geometry_size+num_categories] > 0

        box_nw = (w/2*(1 + r[0] - r[2]/2), h/2*(1 + r[1] - r[3]/2))
        box_se = (w/2*(1 + r[0] + r[2]/2), h/2*(1 + r[1] + r[3]/2))

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