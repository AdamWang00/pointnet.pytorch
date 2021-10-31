import sys
sys.path.insert(0, "/home/awang156/DeepSDF")
import numpy as np
import torch
import os
import json
from pointnetae.model import PointNetAE
from pointnetae.config import *
from pointnetae.utils import *
from pointnetae.dataset import SceneDataset

import deep_sdf
from networks.deep_sdf_decoder_color import Decoder
from deep_sdf.mesh_color import create_mesh

# ========== BEGIN PARAMS ==========

NUM_RECONSTRUCTIONS = 8
DATASET_OFFSET = 0

# THESE MUST REFERENCE THE MODELS WHOSE LATENT CODES ARE USED DURING PREPROCESSING
deepsdf_model_spec_subpaths = {
    "bed": "bed1/specs.json",
    "nightstand": "nightstand1/specs.json"
}
deepsdf_model_param_subpaths = {
    "bed": "bed1/ModelParameters/1000.pth",
    "nightstand": "nightstand1/ModelParameters/1000.pth"
}

deepsdf_experiments_dir = "../../DeepSDF"
deepsdf_experiments_dir = os.path.join(deepsdf_experiments_dir, "experiments")

# ========== END PARAMS ==========

base_dir = os.path.join(data_dir, room_name)
rooms_dir = os.path.join(base_dir, rooms_subdir)

with open(os.path.join(base_dir, "categories.json"), "r") as f:
    categories_reverse_dict = json.load(f)

decoders = []
for idx in range(len(categories_reverse_dict)):
    category = categories_reverse_dict[str(idx)]
    specs_filename = os.path.join(deepsdf_experiments_dir, deepsdf_model_spec_subpaths[category])
    specs = json.load(open(specs_filename))
    latent_size = specs["CodeLength"]
    if specs["NetworkArch"] == "deep_sdf_decoder_color":
        decoder = Decoder(latent_size, **specs["NetworkSpecs"])
    else:
        raise Exception("unrecognized deepsdf decoder arch")
    decoder = torch.nn.DataParallel(decoder)
    params_filename = os.path.join(deepsdf_experiments_dir, deepsdf_model_param_subpaths[category])
    saved_model_state = torch.load(params_filename)
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    decoders.append(decoder)

load_path = os.path.join("experiments", model_name, model_params_subdir, epoch_load + ".pth")
model = PointNetAE()
model.load_state_dict(torch.load(load_path))
model = model.eval().cuda()

scene_dataset = SceneDataset(rooms_dir, max_num_points)

reconstructions_dir = os.path.join("experiments", model_name, model_reconstructions_subdir, epoch_load)
os.makedirs(reconstructions_dir, exist_ok=True)

for i in range(DATASET_OFFSET, DATASET_OFFSET + NUM_RECONSTRUCTIONS):
    os.makedirs(os.path.join(reconstructions_dir, str(i)), exist_ok=True)
    furniture_infos_filepath = os.path.join(reconstructions_dir, str(i), "info.json")

    if os.path.isfile(furniture_infos_filepath):
        print("Room #" + str(i) + " already exists, skipping")
        continue

    print("Reconstructing room #" + str(i))
    scene, target = scene_dataset.__getitem__(i)
    scene = scene.transpose(1, 0).cuda()
    cats = target[:, geometry_size + orientation_size].numpy().astype(int)

    reconstruction, latent_code = model(scene.unsqueeze(0), np.expand_dims(cats, 0))
    reconstruction = reconstruction[0].detach().cpu()
    latent_code = latent_code[0]

    areas = reconstruction[:, 2] * reconstruction[:, 3]
    indices_area_descending = np.argsort(-areas)
    reconstruction = reconstruction[indices_area_descending]

    furniture_info_list = []
    for idx, r in zip(indices_area_descending.tolist(), reconstruction.tolist()):
        pos = r[0:2]
        dim = r[2:4]
        ori = r[4:6]
        ori = clip_orientation(ori / np.linalg.norm(ori))
        cat_idx = np.argmax(r[geometry_size+orientation_size:geometry_size+orientation_size+num_categories])
        cat = categories_reverse_dict[str(cat_idx)]
        existence = r[geometry_size+orientation_size+num_categories] > 0

        if not existence:
            continue

        print(f"Reconstructing room #{i} furniture #{len(furniture_info_list)} (category {cat})")

        mesh_filepath = os.path.join(reconstructions_dir, str(i), "mesh_" + str(len(furniture_info_list)))

        decode_shape_input = torch.cat(
            (
                latent_code,
                reconstruction[idx, 0:geometry_size+orientation_size].cuda()
            )
        )
        shape_code = model.decode_shape(decode_shape_input, cat_idx)

        with torch.no_grad():
            create_mesh(
                decoders[cat_idx],
                shape_code,
                mesh_filepath,
                N=512,
                max_batch=int(2 ** 17),
            )

        # THIS IS DONE IN visualize_reconstruction.py
        # if (ori[1] == 0): # Need to flip dimensions if oriented towards East/West
        #     dim[0], dim[1] = dim[1], dim[0]

        # box_nw = (w/2*(1 + (pos[0] - dim[0]/2)), h/2*(1 + (pos[1] - dim[1]/2)))
        # box_se = (w/2*(1 + (pos[0] + dim[0]/2)), h/2*(1 + (pos[1] + dim[1]/2)))

        # box_center = ((box_nw[0] + box_se[0])/2, (box_nw[1] + box_se[1])/2)

        furniture_info = {
            "mesh_filepath": mesh_filepath + ".ply",
            "pos": pos,
            "dim": dim,
            "ori": ori.tolist(),
            "cat": int(cat_idx),
        }
        furniture_info_list.append(furniture_info)

    with open(furniture_infos_filepath, "w") as f:
        json.dump(furniture_info_list, f)