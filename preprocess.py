import os
import json
import numpy as np
import pickle


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


scenes_dir = "/home/awang/projects/data/3D-FRONT"
data_dir = "/home/awang/projects/pointnet.pytorch/data"
model_info_filepath = "/home/awang/projects/data/3D-FUTURE-model/model_info.json"
rooms_subdir = "Rooms"

room_type = "Bedroom"
furniture_super_categories = {"Bed"} # super-category in 3D-FUTURE-model/model_info.json
furniture_categories = {"Nightstand"} # category in 3D-FUTURE-model/model_info.json

category_maps_dir = os.path.join(data_dir, room_type)
if not os.path.isdir(category_maps_dir):
        os.makedirs(category_maps_dir)

rooms_dir = os.path.join(data_dir, room_type, rooms_subdir)
if not os.path.isdir(rooms_dir):
    os.makedirs(rooms_dir)

categories_dict = {}
categories_reverse_dict = {}
num_categories = 0
for idx, category in enumerate(furniture_super_categories.union(furniture_categories)):
    categories_dict[category] = idx
    categories_reverse_dict[idx] = category
    num_categories += 1

with open(os.path.join(category_maps_dir, "categories.pkl"), "wb") as f:
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

            # if furniture["category"] != model["category"]:
            #     print("Categories do not match: ", furniture["category"], "AND", model["category"])

            category = None
            if model["super-category"] in furniture_super_categories:
                category = model["super-category"]
            elif model["category"] in furniture_categories:
                category = model["category"]
            if category is not None:
                furniture["category"] = category # overwrite
                furniture_dict[furniture["uid"]] = furniture

    room_count_scene = 0 # count of rooms in this scene with type == room_type and at least 1 furniture in furniture_dict
    for room in scene_json["scene"]["room"]:
        if room["type"] == room_type:
            furniture_list = []
            for child in room["children"]: # parts of the room, includes layout and furniture
                ref = child["ref"]
                if ref in furniture_dict:
                    furniture = furniture_dict[ref]
                    pos = child["pos"]
                    dim = np.array(child["scale"]) * np.array(furniture["bbox"])
                    ori = quaternion_to_orientation(child["rot"])
                    cat = categories_dict[furniture["category"]]

                    f = np.zeros(6 + num_categories)
                    f[0:6] = [pos[0], pos[2], dim[0], dim[2], ori[0], ori[2]]
                    f[6 + cat] = 1
                    furniture_list.append(f)
            
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

            room_filename = os.path.splitext(scene_filename)[0] + "_" + str(room_count_scene) + ".npy"
            with open(os.path.join(rooms_dir, room_filename), "wb") as f:
                np.save(f, furniture_arr)

    room_count += room_count_scene

print("Total rooms:", room_count)
print("Average furniture count:", furniture_count/room_count)
print("Max furniture count:", max_furniture_count)