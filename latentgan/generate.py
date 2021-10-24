import os
import torch
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from latentgan.config import *
from latentgan.model import WGAN_GP
if ae_model_class == "pointnetvae":
    from pointnetvae.model import PointNetVAE
elif ae_model_class == "pointnetae":
    from pointnetae.model import PointNetAE
from pointnetvae.utils import clip_orientation

NUM_GENERATIONS = 16
HIDE_NONEXISTENT_OUTPUTS = True

base_dir = os.path.join(data_dir, room_name)
with open(os.path.join(base_dir, "categories.json"), "r") as f:
    categories_reverse_dict = json.load(f)

font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

def draw_generated_scene(generated_scene):
    w, h = 500, 500
    scale = 0.2

    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w, h/2)], fill="gray")

    generated_scene = generated_scene.squeeze().detach().cpu()
    areas = generated_scene[:, 2] * generated_scene[:, 3]
    indices_area_descending = np.argsort(-areas)
    generated_scene = generated_scene[indices_area_descending]
    for idx, r in enumerate(generated_scene.squeeze().tolist()):
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

if __name__ == "__main__":
    ae_load_path = os.path.join("..", ae_model_class, "experiments", ae_model_name, model_params_subdir, ae_epoch_load + ".pth")
    gan_load_path = os.path.join("experiments", model_name, model_params_subdir)

    if ae_model_class == "pointnetvae":
        model_ae = PointNetVAE()
    elif ae_model_class == "pointnetae":
        model_ae = PointNetAE()
    model_ae.load_state_dict(torch.load(ae_load_path))
    model_ae = model_ae.cuda().eval()

    model_gan = WGAN_GP()
    model_gan.load_model(gan_load_path, iter_load)
    model_gan.eval()

    for _ in range(NUM_GENERATIONS):
        latent_code_sample = model_gan.generate()
        generated_scene = model_ae.generate(latent_code=latent_code_sample)
        draw_generated_scene(generated_scene)