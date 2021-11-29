import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
from PIL import Image
import json
import os
from latentgan.config import *

NUM_GENERATIONS = 8
OFFSET = 0

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

for i in range(OFFSET, OFFSET + NUM_GENERATIONS):
    print(i)
    furniture_info_list_generation_path = os.path.join("experiments", model_name, model_generations_subdir, iter_load, str(i), "info.json")

    with open(furniture_info_list_generation_path, "r") as f:
        furniture_info_list_generation = json.load(f)

    scene = Scene()

    # Reconstruction
    for furniture_info_generation in furniture_info_list_generation:
        # print(json.dumps(furniture_info_generation, indent=2))
        mesh_filepath = furniture_info_generation["mesh_filepath"]
        pos = furniture_info_generation["pos"]
        dim = furniture_info_generation["dim"]
        ori = furniture_info_generation["ori"]
        # cat = furniture_info_generation["cat"]

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
            angle = np.arctan(np.divide(ori[0], ori[1] + 1e-8))
            gen_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, y_axis))

            # translate
            gen_mesh.apply_translation((pos[0], scale_y * bbox_dim[1] / 2, pos[1])) # todo: use y pos

            scene.add_node(Node(mesh=Mesh.from_trimesh(gen_mesh), translation=[0, 0, 0]))
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