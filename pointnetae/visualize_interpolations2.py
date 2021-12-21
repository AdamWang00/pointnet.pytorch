import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
import json
import os
import time
from pointnetae.config import *

IS_TESTING = True
ROOM_IDX_1 = 4
ROOM_IDX_2 = 44
FPS = 4

SAVE_GIF = False
SAVE_GIF_PATH = "/home/awang/Desktop/interpolation.gif"

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

if IS_TESTING:
    model_interpolations_subdir = model_testing_interpolations_subdir
else:
    model_interpolations_subdir = model_training_interpolations_subdir
interpolations_dir = os.path.join("experiments", model_name, model_interpolations_subdir, epoch_load, f"{ROOM_IDX_1}_to_{ROOM_IDX_2}")

scene = Scene()

node_lists = [] # contains lists (interpolated scenes) of Nodes (furniture)
for i in range(1337): # we do not know the number of interpolations, so we loop through each i until invalid
    furniture_info_list_interpolation_path = os.path.join(interpolations_dir, str(i), "info.json")
    if not os.path.isfile(furniture_info_list_interpolation_path):
        break

    print(i)

    with open(furniture_info_list_interpolation_path, "r") as f:
        furniture_info_list_interpolation = json.load(f)

    mesh_node_list = []
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

            mesh_node = Node(mesh=Mesh.from_trimesh(gen_mesh, smooth=False))
            mesh_node.mesh.is_visible = False
            mesh_node_list.append(mesh_node)
            scene.add_node(mesh_node)
        except ValueError as e:
            print("[error]", str(e))
            continue
    
    node_lists.append(mesh_node_list)

camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_w/viewport_h)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 4.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

if SAVE_GIF:
    viewer = Viewer(scene, use_raymond_lighting=True, viewport_size=(viewport_w,viewport_h), render_flags={"cull_faces": False}, run_in_thread=True, record=True)

    viewer.render_lock.acquire()
    for mesh_node in node_lists[0]:
        mesh_node.mesh.is_visible = True
    viewer.render_lock.release()

    time.sleep(1)
    for i in range(1, len(node_lists)):
        viewer.render_lock.acquire()
        for mesh_node in node_lists[i - 1]:
            mesh_node.mesh.is_visible = False
        for mesh_node in node_lists[i]:
            mesh_node.mesh.is_visible = True
        viewer.render_lock.release()
        time.sleep(1/FPS)
    time.sleep(1)

    viewer.close_external()
    viewer.save_gif(SAVE_GIF_PATH)
else:
    viewer = Viewer(scene, use_raymond_lighting=True, viewport_size=(viewport_w,viewport_h), render_flags={"cull_faces": False}, run_in_thread=True)

    while True:
        viewer.render_lock.acquire()
        for mesh_node in node_lists[-1]:
            mesh_node.mesh.is_visible = False
        for mesh_node in node_lists[0]:
            mesh_node.mesh.is_visible = True
        viewer.render_lock.release()

        time.sleep(1)
        for i in range(1, len(node_lists)):
            viewer.render_lock.acquire()
            for mesh_node in node_lists[i - 1]:
                mesh_node.mesh.is_visible = False
            for mesh_node in node_lists[i]:
                mesh_node.mesh.is_visible = True
            viewer.render_lock.release()
            time.sleep(1/FPS)
        time.sleep(1)