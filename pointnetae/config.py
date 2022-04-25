import os

model_name = "bedroom_ori2"
epoch_load = "latest"
print("pointnetae", model_name, epoch_load)

data_dir = "../data"
split_dir = "../splits"
rooms_subdir = "Rooms"
roominfos_subdir = "RoomInfos"
gt_models_path = "../../data/3D-FUTURE-model/all"

model_params_subdir = "ModelParameters"
model_training_reconstructions_subdir = "TrainingReconstructions"
model_testing_reconstructions_subdir = "TestingReconstructions"
model_training_interpolations_subdir = "TrainingInterpolations"
model_testing_interpolations_subdir = "TestingInterpolations"

params_history = {
    "bedroom_full2a":{ # bedroom_full1c1
        "procedure": "bedroom2",
        "split_train": "test64/train.json",
        "split_test": "test64/test.json",
        "num_epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 512,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_full2b":{ # bedroom_full1c1
        "procedure": "bedroom2",
        "split_train": "test64/train.json",
        "split_test": "test64/test.json",
        "num_epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 512,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_full2c":{ # bedroom_full1c1
        "procedure": "bedroom2",
        "split_train": "test64/train.json",
        "split_test": "test64/test.json",
        "num_epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 512,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_ori1": { # bedroom_full2b
        "procedure": "bedroom3",
        "split_train": "test64/train.json",
        "split_test": "test64/test.json",
        "num_epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 512,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_ori1a": { # bedroom_full2b
        "procedure": "bedroom3",
        "split_train": "test64/train.json",
        "split_test": "test64/test.json",
        "num_epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 256,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_ori2": {
        "procedure": "bedroom4",
        "split_train": "test64/train.json",
        "split_test": "test64/test.json",
        "num_epochs": 1000,
        "batch_size": 64,
        "learning_rate": 0.0002,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 512,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
}

procedure_params_all = {
    "bedroom1": {
        "room_name": "Bedroom1",
        "geometry_size": 2 + 2, # position and dimensions
        "orientation_size": 2,
        "num_categories": 2,
        "shape_size": 512,
    },
    "bedroom2": {
        "room_name": "Bedroom2",
        "geometry_size": 2 + 2, # position and dimensions
        "orientation_size": 2,
        "num_categories": 9,
        "shape_size": 512,
    },
    "bedroom3": {
        "room_name": "Bedroom3",
        "geometry_size": 2 + 2, # position and dimensions
        "orientation_size": 2,
        "num_categories": 9,
        "shape_size": 512,
    },
    "bedroom4": {
        "room_name": "Bedroom4",
        "deepsdf_ver": 2,
        "position_size": 3,
        "dimension_size": 3,
        "orientation_size": 2,
        "num_categories": 10,
        "shape_size": 512,
    }
}

colors = {
    "bed": "blue",
    "cabinet": "grey",
    "chair": "red",
    "smallStool": "red",
    "largeSofa": "yellow",
    "largeTable": "brown",
    "nightstand": "orange",
    "tvStand": "green",
    "smallTable": "purple",
    "lighting": "yellow",
}

colors_light = {
    "bed": "#ADD8EE",
    "cabinet": "#D3D3D3",
    "nightstand": "#FFB580",
    "chair": "#FF7F7F",
    "smallStool": "#FF7F7F",
    "largeSofa": "#FFFF99",
    "largeTable": "#C89D7C",
    "tvStand": "#B0EEB0",
    "smallTable": "#b19cd9",
    "lighting": "#FFFF99",
}

params = params_history[model_name]
procedure = params["procedure"]
procedure_params = procedure_params_all[procedure]

num_examples = params["num_examples"] if "num_examples" in params else None
num_epochs = params["num_epochs"]
batch_size = params["batch_size"]
learning_rate = params["learning_rate"]
step_size = params["step_size"]
step_gamma = params["step_gamma"]

latent_size = params["latent_size"]
max_num_points = params["max_num_points"]

room_name = procedure_params["room_name"]
position_size = procedure_params["position_size"] if "position_size" in procedure_params else 2
dimension_size = procedure_params["dimension_size"] if "dimension_size" in procedure_params else 2
geometry_size = procedure_params["geometry_size"] if "geometry_size" in procedure_params else position_size + dimension_size
orientation_size = procedure_params["orientation_size"]
num_categories = procedure_params["num_categories"]
shape_size = procedure_params["shape_size"]

point_size_intermediate = geometry_size + orientation_size + num_categories
point_size = point_size_intermediate + shape_size

geometric_weight = params["geometric_weight"]
orientation_weight = params["orientation_weight"]
categorical_weight = params["categorical_weight"]
existence_weight = params["existence_weight"]
shape_weight = params["shape_weight"]

dimensions_matching_weight = params["dimensions_matching_weight"]

encoder_hidden_dims = params["encoder_hidden_dims"]
decoder_hidden_dims = params["decoder_hidden_dims"]
shape_code_encoder_hidden_dims = params["shape_code_encoder_hidden_dims"]
shape_code_encoder_output_size = params["shape_code_encoder_output_size"]
shape_code_decoder_hidden_dims = params["shape_code_decoder_hidden_dims"]

split_train = params["split_train"]
split_test = params["split_test"]

deepsdf_ver = procedure_params["deepsdf_ver"] if "deepsdf_ver" in procedure_params else 1
deepsdf_model_spec_subpaths = {
    "bed": "bed"+str(deepsdf_ver)+"/specs.json",
    "cabinet": "cabinet"+str(deepsdf_ver)+"/specs.json",
    "chair": "chair"+str(deepsdf_ver)+"/specs.json",
    "largeSofa": "largeSofa"+str(deepsdf_ver)+"/specs.json",
    "largeTable": "largeTable"+str(deepsdf_ver)+"/specs.json",
    "nightstand": "nightstand"+str(deepsdf_ver)+"/specs.json",
    "smallStool": "smallStool"+str(deepsdf_ver)+"/specs.json",
    "smallTable": "smallTable"+str(deepsdf_ver)+"/specs.json",
    "tvStand": "tvStand"+str(deepsdf_ver)+"/specs.json",
    "lighting": "lighting"+str(deepsdf_ver)+"/specs.json",
}
deepsdf_model_param_subpaths = {
    "bed": "bed"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "cabinet": "cabinet"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "chair": "chair"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "largeSofa": "largeSofa"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "largeTable": "largeTable"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "nightstand": "nightstand"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "smallStool": "smallStool"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "smallTable": "smallTable"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "tvStand": "tvStand"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
    "lighting": "lighting"+str(deepsdf_ver)+"/ModelParameters/1000.pth",
}

deepsdf_experiments_dir = "../../DeepSDF"
deepsdf_experiments_dir = os.path.join(deepsdf_experiments_dir, "experiments")