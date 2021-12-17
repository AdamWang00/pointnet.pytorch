import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
from PIL import Image
import json
import os
from pointnetae.config import *
from pointnetae.dataset import SceneDataset

IS_TESTING = True
NUM_RECONSTRUCTIONS = 64
DATASET_OFFSET = 0

viewport_w = 900
viewport_h = 900

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

scene_dataset = SceneDataset(rooms_dir, max_num_points, is_testing=IS_TESTING)

for i in range(DATASET_OFFSET, DATASET_OFFSET + NUM_RECONSTRUCTIONS):
    room_id = scene_dataset.get_room_id(i)
    print(i, room_id)

    furniture_info_list_gt_path = os.path.join(roominfos_dir, room_id + ".json")
    with open(furniture_info_list_gt_path, "r") as f:
        furniture_info_list_gt = json.load(f)

    scene = Scene()

    # GT
    for furniture_info_gt in furniture_info_list_gt:
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
            angle = np.arctan2(ori[0], ori[1])
            gt_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, y_axis))

            # translate
            gt_mesh.apply_translation((pos[0], scale_y * bbox_dim[1] / 2, pos[1])) # todo: use y pos

            scene.add_node(Node(mesh=Mesh.from_trimesh(gt_mesh, smooth=False)))
        except ValueError as e:
            print("[error]", str(e))
            continue

    camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_w/viewport_h)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 8.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    Viewer(scene, use_raymond_lighting=True, viewport_size=(viewport_w,viewport_h), render_flags={"cull_faces": False})