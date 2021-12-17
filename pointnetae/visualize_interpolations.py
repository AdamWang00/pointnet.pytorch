import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
import json
import os
from pointnetae.config import *

IS_TESTING = True
ROOM_IDX_1 = 4
ROOM_IDX_2 = 44

viewport_w = 1800
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

if IS_TESTING:
    model_interpolations_subdir = model_testing_interpolations_subdir
else:
    model_interpolations_subdir = model_training_interpolations_subdir
interpolations_dir = os.path.join("experiments", model_name, model_interpolations_subdir, epoch_load, f"{ROOM_IDX_1}_to_{ROOM_IDX_2}")

scene = Scene()
gap_size = 4

for i in range(1337): # we do not know the number of interpolations, so we loop through each i until invalid
    print(i)
    furniture_info_list_interpolation_path = os.path.join(interpolations_dir, str(i), "info.json")
    if not os.path.isfile(furniture_info_list_interpolation_path):
        break

    with open(furniture_info_list_interpolation_path, "r") as f:
        furniture_info_list_interpolation = json.load(f)

    for furniture_info_interpolation in furniture_info_list_interpolation:
        # print(json.dumps(furniture_info_interpolation, indent=2))
        mesh_filepath = furniture_info_interpolation["mesh_filepath"]
        pos = furniture_info_interpolation["pos"]
        dim = furniture_info_interpolation["dim"]
        ori = furniture_info_interpolation["ori"]
        # cat = furniture_info_interpolation["cat"]

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
            angle = np.arctan2(ori[0], ori[1])
            gen_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, y_axis))

            # translate
            gen_mesh.apply_translation((pos[0], scale_y * bbox_dim[1] / 2, pos[1])) # todo: use y pos

            scene.add_node(Node(mesh=Mesh.from_trimesh(gen_mesh, smooth=False), translation=[i * gap_size, 0, 0]))
        except ValueError as e:
            print("[error]", str(e))
            continue

camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_w/viewport_h)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 4.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

Viewer(scene, use_raymond_lighting=True, viewport_size=(viewport_w,viewport_h), render_flags={"cull_faces": False})