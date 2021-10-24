import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
from PIL import Image
import json
import os
from pointnetae.config import *
from pointnetae.dataset import SceneDataset

NUM_RECONSTRUCTIONS = 1
DATASET_OFFSET = 0

def get_trimesh_and_uv(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(g for g in scene_or_mesh.geometry.values())
        )
        uv = np.concatenate(
            tuple(g.visual.uv for g in scene_or_mesh.geometry.values()),
            axis=0
        )
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
        uv = mesh.visual.uv
    return mesh, uv

base_dir = os.path.join(data_dir, room_name)
rooms_dir = os.path.join(base_dir, rooms_subdir)
roominfos_dir = os.path.join(base_dir, roominfos_subdir)

scene_dataset = SceneDataset(rooms_dir, max_num_points)

for i in range(DATASET_OFFSET, DATASET_OFFSET + NUM_RECONSTRUCTIONS):
    room_id = scene_dataset.get_room_id(i)

    furniture_info_list_gt_path = os.path.join(roominfos_dir, room_id + ".json")
    furniture_info_list_reconstruction_path = os.path.join("experiments", model_name, "TrainingReconstructions", epoch_load, str(i), "info.json")

    with open(furniture_info_list_gt_path, "r") as f:
        furniture_info_list_gt = json.load(f)
    
    with open(furniture_info_list_reconstruction_path, "r") as f:
        furniture_info_list_reconstruction = json.load(f)

    scene = Scene()
    gap_size = 2

    for furniture_info_gt in furniture_info_list_gt[0:2]:
        # print(json.dumps(furniture_info_gt, indent=2))
        model_id = furniture_info_gt["id"]
        pos = furniture_info_gt["pos"]
        dim = furniture_info_gt["dim"]
        ori = furniture_info_gt["ori"]
        # cat = furniture_info_gt["cat"]

        gt_path = os.path.join('../../data/3D-FUTURE-model/all', model_id, 'normalized_model.obj')
        gt_texture_path = os.path.join('../../data/3D-FUTURE-model/all', model_id, 'texture.png')
        
        try:
            gt_mesh, gt_uv = get_trimesh_and_uv(trimesh.load(gt_path, process=False))
            texture_im = Image.open(gt_texture_path)
            gt_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=gt_uv, image=texture_im).to_color()

            # scale
            bbox_dim = gt_mesh.bounding_box.extents
            scale_x = dim[0] / bbox_dim[0]
            scale_z = dim[1] / bbox_dim[2]
            scale_y = (scale_x + scale_z) / 2 # todo: use y dim
            gt_mesh.apply_scale((scale_x, scale_y, scale_z))

            # rotate
            y_axis = [0, 1, 0]
            angle = np.arctan(np.divide(ori[0], ori[1]))
            gt_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, y_axis))

            # translate
            gt_mesh.apply_translation((pos[0], 0, pos[1])) # todo: use y pos

            scene.add_node(Node(mesh=Mesh.from_trimesh(gt_mesh), translation=[-gap_size, 0, 0]))
        except ValueError as e:
            print("[error]", str(e))
            continue

    for furniture_info_reconstruction in furniture_info_list_reconstruction:
        print(json.dumps(furniture_info_reconstruction, indent=2))
        mesh_filepath = furniture_info_reconstruction["mesh_filepath"]
        pos = furniture_info_reconstruction["pos"]
        dim = furniture_info_reconstruction["dim"]
        ori = furniture_info_reconstruction["ori"]
        # cat = furniture_info_reconstruction["cat"]

        try:
            gen_mesh = trimesh.load(mesh_filepath, process=False)
            assert gen_mesh.visual.kind == 'vertex'
            
            # scale
            bbox_dim = gen_mesh.bounding_box.extents
            scale_x = dim[0] / bbox_dim[0]
            scale_z = dim[1] / bbox_dim[2]
            scale_y = (scale_x + scale_z) / 2 # todo: use y dim
            gen_mesh.apply_scale((scale_x, scale_y, scale_z))

            # rotate
            y_axis = [0, 1, 0]
            angle = np.arctan(np.divide(ori[0], ori[1]))
            gen_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, y_axis))

            # translate
            gen_mesh.apply_translation((pos[0], 0, pos[1])) # todo: use y pos

            scene.add_node(Node(mesh=Mesh.from_trimesh(gen_mesh), translation=[gap_size, 0, 0]))
        except ValueError as e:
            print("[error]", str(e))
            continue

    camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    Viewer(scene, use_raymond_lighting=True, viewport_size=(1200,900))