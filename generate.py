import torch
from pointnet.model import PointNetVAE
from pointnet.config import *
import numpy as np
from PIL import Image, ImageDraw

LOAD_PATH = "experiments/model1/latest.pth"
model = PointNetVAE()
model.load_state_dict(torch.load(LOAD_PATH))
model = model.eval()

generated_scene = model.generate()

w, h = 500, 500

img = Image.new("RGB", (w, h), color="white")
draw = ImageDraw.Draw(img)

draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
draw.rectangle([(0, h/2), (w, h/2)], fill="gray")

for r in generated_scene.squeeze().tolist():
    category = np.argmax(r[geometry_size:geometry_size+num_classes])
    existence = r[geometry_size+num_classes] > 0
    if existence:
        draw.rectangle(
            [
                (w/2*(1 + r[0] - r[2]/2), h/2*(1 + r[1] - r[3]/2)),
                (w/2*(1 + r[0] + r[2]/2), h/2*(1 + r[1] + r[3]/2))
            ],
            fill=("blue" if category == 0 else "orange"),
            width=3,
            outline="black"
        )
    else:
        draw.rectangle(
            [
                (w/2*(1 + r[0] - r[2]/2), h/2*(1 + r[1] - r[3]/2)),
                (w/2*(1 + r[0] + r[2]/2), h/2*(1 + r[1] + r[3]/2))
            ],
            fill=("#dbf0fe" if category == 0 else "#ffeeda"),
            width=1,
            outline="black"
        )

img.show()