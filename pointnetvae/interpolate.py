import torch
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pointnetvae.model import PointNetVAE
from pointnetvae.config import *

LOAD_PATH = os.path.join("experiments", model_name, model_params_subdir, epoch_load + ".pth")
NUM_INTERPOLATIONS = 5

# latent code goes from LATENT2 => LATENT1, so most recently generated will be LATENT1
LATENT1 = torch.Tensor([0, 0, 0, -2])
LATENT2 = torch.Tensor([0, 0, 0, 2])

model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

for i in range(NUM_INTERPOLATIONS):
    latent1_weight = i / (NUM_INTERPOLATIONS - 1)
    latent2_weight = 1 - latent1_weight
    latent_code = LATENT1 * latent1_weight + LATENT2 * latent2_weight
    print(latent_code)
    generated_scene = model.generate(latent_code=latent_code.unsqueeze(0))

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