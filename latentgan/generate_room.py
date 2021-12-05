import sys

from pointnetae.reconstruct_room import ORI_CLIP_THRESHOLD
sys.path.insert(0, "/home/awang156/DeepSDF")
import os
import torch
import json
import numpy as np
from latentgan.config import *
from latentgan.model import WGAN_GP
if ae_model_class == "pointnetvae":
    from latentgan.model import PointNetVAE
elif ae_model_class == "pointnetae":
    from latentgan.model import PointNetAE

import deep_sdf
from networks.deep_sdf_decoder_color import Decoder
from deep_sdf.mesh_color import create_mesh

# ========== BEGIN PARAMS ==========

NUM_GENERATIONS = 8
ORI_CLIP_THRESHOLD = 0.8

# THESE MUST REFERENCE THE MODELS WHOSE LATENT CODES ARE USED DURING PREPROCESSING
deepsdf_model_spec_subpaths = {
    "bed": "bed1/specs.json",
    "cabinet": "cabinet1/specs.json",
    "chair": "chair1/specs.json",
    "largeSofa": "largeSofa1/specs.json",
    "largeTable": "largeTable1/specs.json",
    "nightstand": "nightstand1/specs.json",
    "smallStool": "smallStool1/specs.json",
    "smallTable": "smallTable1/specs.json",
    "tvStand": "tvStand1/specs.json",
}
deepsdf_model_param_subpaths = {
    "bed": "bed1/ModelParameters/1000.pth",
    "cabinet": "cabinet1/ModelParameters/1000.pth",
    "chair": "chair1/ModelParameters/1000.pth",
    "largeSofa": "largeSofa1/ModelParameters/1000.pth",
    "largeTable": "largeTable1/ModelParameters/1000.pth",
    "nightstand": "nightstand1/ModelParameters/1000.pth",
    "smallStool": "smallStool1/ModelParameters/1000.pth",
    "smallTable": "smallTable1/ModelParameters/1000.pth",
    "tvStand": "tvStand1/ModelParameters/1000.pth",
}

deepsdf_experiments_dir = "../../DeepSDF"
deepsdf_experiments_dir = os.path.join(deepsdf_experiments_dir, "experiments")

# ========== END PARAMS ==========

base_dir = os.path.join(data_dir, room_name)
with open(os.path.join(base_dir, "categories.json"), "r") as f:
    categories_reverse_dict = json.load(f)

def clip_orientation(d, threshold=0.0):
    '''
    clip to [+-1, +-1] if close enough
    '''
    major_orientations = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ]
    max_index = -1
    max_dot = 0
    for idx, major_orientation in enumerate(major_orientations):
        dot = np.dot(d, major_orientation)
        if dot > max_dot:
            max_dot = dot
            max_index = idx
    if max_dot > threshold:
        return major_orientations[max_index]
    else:
        return d


if __name__ == "__main__":
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

    ae_load_path = os.path.join("..", ae_model_class, "experiments", ae_model_name, model_params_subdir, ae_epoch_load + ".pth")
    gan_load_path = os.path.join("experiments", model_name, model_params_subdir)

    if ae_model_class == "pointnetvae":
        model_ae = PointNetVAE()
    elif ae_model_class == "pointnetae":
        model_ae = PointNetAE()
    model_ae.load_state_dict(torch.load(ae_load_path))
    model_ae = model_ae.eval().cuda()

    model_gan = WGAN_GP()
    model_gan.load_model(gan_load_path, iter_load)
    model_gan.eval()

    generations_dir = os.path.join("experiments", model_name, model_generations_subdir, iter_load)
    os.makedirs(generations_dir, exist_ok=True)

    for i in range(NUM_GENERATIONS):
        os.makedirs(os.path.join(generations_dir, str(i)), exist_ok=True)
        furniture_infos_filepath = os.path.join(generations_dir, str(i), "info.json")

        if os.path.isfile(furniture_infos_filepath):
            print("Room #" + str(i) + " already exists, skipping")
            continue

        print("Generating room #" + str(i))

        latent_code = model_gan.generate()
        generation = model_ae.generate(latent_code=latent_code)
        latent_code = latent_code[0]
        generation = generation.detach().cpu()

        areas = generation[:, 2] * generation[:, 3]
        indices_area_descending = np.argsort(-areas)
        generation = generation[indices_area_descending]

        furniture_info_list = []
        for idx, r in zip(indices_area_descending.tolist(), generation.tolist()):
            pos = r[0:2]
            dim = r[2:4]
            ori = r[4:6]
            ori = clip_orientation(ori / np.linalg.norm(ori), threshold=ORI_CLIP_THRESHOLD)
            cat_idx = np.argmax(r[geometry_size+orientation_size:geometry_size+orientation_size+num_categories])
            cat = categories_reverse_dict[str(cat_idx)]
            existence = r[geometry_size+orientation_size+num_categories] > 0

            if not existence:
                continue

            print(f"Generating room #{i} furniture #{len(furniture_info_list)} (category {cat})")

            mesh_filepath = os.path.join(generations_dir, str(i), "mesh_" + str(len(furniture_info_list)))

            decode_shape_input = torch.cat(
                (
                    latent_code,
                    generation[idx, 0:geometry_size+orientation_size].cuda()
                )
            )
            shape_code = model_ae.decode_shape(decode_shape_input, cat_idx)

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