import os
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont

data_dir = "./data"
rooms_subdir = "Rooms"

room_type = "Bedroom"

colors = {
    "Bed": "blue",
    "Nightstand": "orange"
}

# font = ImageFont.truetype('/home/awang/Roboto-Regular.ttf', 12)

base_dir = os.path.join(data_dir, room_type)
rooms_dir = os.path.join(base_dir, rooms_subdir)

with open(os.path.join(base_dir, "categories.pkl"), "rb") as f:
    categories_reverse_dict = pickle.load(f)

for room_filename in os.listdir(rooms_dir):
    w, h = 500, 500
    scale = 0.25

    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(w/2, 0), (w/2, h)], fill="gray")
    draw.rectangle([(0, h/2), (w, h/2)], fill="gray")

    furniture_arr = np.load(os.path.join(rooms_dir, room_filename), allow_pickle=True)
    for furniture in furniture_arr:
        pos = furniture[0:2]
        dim = furniture[2:4]
        ori = furniture[4:6]
        cat = categories_reverse_dict[furniture[6]]

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
        print(box_center, ori_indicator)
        draw.line([box_center, ori_indicator], fill='white')

    img.show()