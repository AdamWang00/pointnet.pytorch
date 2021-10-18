import os
import json
import numpy as np
import pickle
import torch
from pointnetvae.config import data_dir, rooms_subdir    


def vector_dot_matrix3(v, mat):
    rot_mat = np.mat(mat)
    vec = np.mat(v).T
    result = np.dot(rot_mat, vec)
    return np.array(result.T)[0]


def clip_orientation(d, threshold=0.8):
    '''
    clip to [+-1, 0, +-1] if close enough
    '''
    major_orientations = [
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
        np.array([1, 0, 0]),
        np.array([-1, 0, 0]),
    ]
    for major_orientation in major_orientations:
        if np.dot(d, major_orientation) > threshold:
            return major_orientation
    return d


def quaternion_to_matrix(args):
    """
    quaternion_to_matrix
    :param args:
    :return:
    """
    tx = args[0] + args[0]
    ty = args[1] + args[1]
    tz = args[2] + args[2]
    twx = tx * args[3]
    twy = ty * args[3]
    twz = tz * args[3]
    txx = tx * args[0]
    txy = ty * args[0]
    txz = tz * args[0]
    tyy = ty * args[1]
    tyz = tz * args[1]
    tzz = tz * args[2]

    result = np.zeros((3, 3))
    result[0, 0] = 1.0 - (tyy + tzz)
    result[0, 1] = txy - twz
    result[0, 2] = txz + twy
    result[1, 0] = txy + twz
    result[1, 1] = 1.0 - (txx + tzz)
    result[1, 2] = tyz - twx
    result[2, 0] = txz - twy
    result[2, 1] = tyz + twx
    result[2, 2] = 1.0 - (txx + tyy)
    return result


def quaternion_to_orientation(qua, axis = np.array([0, 0, 1])):
    """
    Quaternion to orientation, with clipping
    :param qua
    :param axis
    :return:
    """
    rotMatrix = quaternion_to_matrix(qua)
    return clip_orientation(vector_dot_matrix3(axis, rotMatrix))


SAVE = True

room_type = "Bedroom" # name of room in 3D-FRONT
room_name = "Bedroom3" # name of subdirectory to save to

super_categories = {'bed', 'nightstand'}

scenes_dir = "../../data/3D-FRONT"
model_info_filepath = "../../data/model_info.json"

model_category_to_super_category = {'armchair': 'chair', 'Lounge Chair / Cafe Chair / Office Chair': 'chair', 'Pendant Lamp': 'lighting', 'Coffee Table': 'largeTable', 'Corner/Side Table': 'smallTable', 'Dining Table': 'largeTable', 'King-size Bed': 'bed', 'Nightstand': 'nightstand', 'Bookcase / jewelry Armoire': 'cabinet', 'Three-Seat / Multi-set sofa': 'largeSofa', 'TV Stand': 'tvStand', 'Drawer Chest / Corner cabinet': 'cabinet', 'Wardrobe': 'cabinet', 'Footstool / Sofastool / Bed End Stool / Stool': 'smallStool', 'Sideboard / Side Cabinet / Console': 'cabinet', 'Ceiling Lamp': 'lighting', 'Children Cabinet': 'cabinet', 'Bed Frame': 'bed', 'Round End Table': 'smallTable', 'Desk': 'largeTable', 'Single bed': 'bed', 'Loveseat Sofa': 'largeSofa', 'Dining Chair': 'chair', 'Barstool': 'chair', 'Lazy Sofa': 'chair', 'L-shaped Sofa': 'largeSofa', 'Wine Cooler': 'cabinet', 'Dressing Table': 'largeTable', 'Dressing Chair': 'chair', 'Kids Bed': 'bed', 'Classic Chinese Chair': 'chair', 'Bunk Bed': 'bed', 'Chaise Longue Sofa': 'largeSofa', 'Shelf': 'cabinet', '衣帽架': 'other', '脚凳/墩': 'other', '博古架': 'other', '置物架': 'other', '装饰架': 'other'}

deepsdf_experiments_dir = "../../DeepSDF/experiments"
deepsdf_latent_code_subpaths = {
    "bed": "bed1/LatentCodes/1000.pth",
    "nightstand": "nightstand1/LatentCodes/1000.pth"
}
deepsdf_latent_codes = {} # model_id -> np.array
for super_category in super_categories:
    deepsdf_split_path = os.path.join(deepsdf_experiments_dir, "splits", super_category + ".json")
    with open(deepsdf_split_path, "r") as f:
        split = json.load(f)
    deepsdf_latent_code_path = os.path.join(deepsdf_experiments_dir, deepsdf_latent_code_subpaths[super_category])
    latent_codes = torch.load(deepsdf_latent_code_path)["latent_codes"]["weight"].numpy()

    latent_code_idx = 0
    for dataset in split:
        for category in split[dataset]:
            for model_id in split[dataset][category]:
                deepsdf_latent_codes[model_id] = latent_codes[latent_code_idx, :]
                latent_code_idx += 1
    assert latent_code_idx == latent_codes.shape[0]

if SAVE:
    base_dir = os.path.join(data_dir, room_name)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    rooms_dir = os.path.join(base_dir, rooms_subdir)
    if not os.path.isdir(rooms_dir):
        os.makedirs(rooms_dir)

categories_dict = {}
categories_reverse_dict = {}
categories_count_dict_scene = {} # count of number of scenes with this cat
unmatched_categories_count_dict_scene = {} # count of number of scenes with this cat
categories_count_dict_room = {} # total number of furniture with this cat in room_type (used to experiment with dividing/grouping categories)
num_categories = 0
for idx, category in enumerate(super_categories):
    categories_dict[category] = idx
    categories_reverse_dict[idx] = category
    categories_count_dict_scene[category] = 0
    categories_count_dict_room[category] = 0
    num_categories += 1

if SAVE:
    with open(os.path.join(base_dir, "categories.pkl"), "wb") as f:
        pickle.dump(categories_reverse_dict, f, pickle.HIGHEST_PROTOCOL)

with open(model_info_filepath, "r") as f:
    model_info_json = json.load(f)

model_dict = {} # all models in 3D-FUTURE-model
for model in model_info_json:
    model_dict[model["model_id"]] = model

scene_filenames = os.listdir(scenes_dir)
num_scenes = len(scene_filenames)
scene_count = 0
room_count = 0
furniture_count = 0
max_furniture_count = 0

for scene_filename in scene_filenames:
    scene_count += 1
    if scene_filename.endswith(".json"):
        print(f"Preprocessing scene ({scene_count}/{num_scenes}):", scene_filename)
    else:
        print("Skipping scene:", scene_filename)
        continue
    
    with open(os.path.join(scenes_dir, scene_filename), "r") as f:
        scene_json = json.load(f)

    furniture_dict = {} # all furniture in the scene that are in one of the above categories
    for furniture in scene_json["furniture"]:
        if "valid" in furniture and furniture["valid"]:
            jid = furniture["jid"]

            model = model_dict[jid]
            model_super_category = model_category_to_super_category[model["category"]]
            if model_super_category in super_categories:
                categories_count_dict_scene[model_super_category] += 1
                furniture["category"] = model_super_category # overwrite
                furniture_dict[furniture["uid"]] = furniture
            elif model_super_category in unmatched_categories_count_dict_scene:
                unmatched_categories_count_dict_scene[model_super_category] += 1
            else:
                unmatched_categories_count_dict_scene[model_super_category] = 1
                

    room_count_scene = 0 # count of rooms in this scene with type == room_type and at least 1 furniture in furniture_dict
    for room in scene_json["scene"]["room"]:
        if room["type"] == room_type:
            furniture_list = []
            furniture_info_list = []
            for child in room["children"]: # parts of the room, includes layout and furniture
                ref = child["ref"]
                if ref in furniture_dict:
                    furniture = furniture_dict[ref]
                    # ignore models that were not part of deepsdf training, as they do not have latent code
                    if furniture['jid'] not in deepsdf_latent_codes:
                        continue

                    pos = child["pos"]
                    dim = np.array(child["scale"]) * np.array(furniture["bbox"])
                    ori = quaternion_to_orientation(child["rot"])
                    cat = categories_dict[furniture["category"]]
                    categories_count_dict_room[furniture["category"]] += 1
                    shape_code = deepsdf_latent_codes[furniture["jid"]]

                    f = np.zeros(6 + 1 + 512) # 4 geo, 2 ori, 1 cat, 512 shape
                    f[0:6] = [pos[0], pos[2], dim[0], dim[2], ori[0], ori[2]]
                    f[6] = cat
                    f[7:] = shape_code
                    furniture_list.append(f)
                    furniture_info_list.append({
                        "id": furniture['jid'],
                        "pos": [pos[0], pos[2]],
                        "dim": [dim[0], dim[2]],
                        "ori": [ori[0], ori[2]],
                        "cat": cat
                    })
            
            furniture_count_room = len(furniture_list)
            if furniture_count_room == 0:
                continue

            room_count_scene += 1
            furniture_count += furniture_count_room
            max_furniture_count = max(max_furniture_count, furniture_count_room)
            furniture_arr = np.array(furniture_list)

            # center room around average of furniture positions
            pos_avg = np.average(furniture_arr[:, 0:2], axis=0)
            furniture_arr[:, 0:2] -= pos_avg

            if SAVE:
                room_filename = os.path.splitext(scene_filename)[0] + "_" + str(room_count_scene)
                np.save(os.path.join(rooms_dir, room_filename + ".npy"), furniture_arr)
                with open(os.path.join(rooms_dir, room_filename + ".json"), "w") as f:
                    json.dump(furniture_info_list, f)

    room_count += room_count_scene

info = [
    f"Total rooms: {room_count}\n",
    f"Average furniture count: {furniture_count/room_count}\n",
    f"Max furniture count: {max_furniture_count}\n",
    f"Matched categories scene count: {json.dumps(categories_count_dict_scene)}\n",
    f"Unmatched categories scene count: {json.dumps(unmatched_categories_count_dict_scene)}\n",
    f"Matched categories frequencies across rooms: {json.dumps(categories_count_dict_room)}\n",
]

if SAVE:
    with open(os.path.join(base_dir, "info.txt"), "w") as f:
        f.writelines(info)

for message in info:
    print(message, end="")