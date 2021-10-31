model_name = "bedroom_partial6d"
epoch_load = "latest"
print("pointnetae", model_name, epoch_load)

data_dir = "../data"
rooms_subdir = "Rooms"
roominfos_subdir = "RoomInfos"

model_params_subdir = "ModelParameters"
model_reconstructions_subdir = "TrainingReconstructions"

params_history = {
    "bedroom_partial3": {
        "procedure": "bedroom1",
        "num_epochs": 800,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 200,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_partial4": {
        "procedure": "bedroom1",
        "num_epochs": 800,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 200,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 10,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_partial5": {
        "procedure": "bedroom1",
        "num_epochs": 800,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 200,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 100,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_partial6": {
        "procedure": "bedroom1",
        "num_epochs": 800,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 200,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_partial6a": {
        "procedure": "bedroom1",
        "num_epochs": 1001,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512, 512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_partial6b": {
        "procedure": "bedroom1",
        "num_epochs": 1001,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512, 512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512],
    },
    "bedroom_partial6c": {
        "procedure": "bedroom1",
        "num_epochs": 1001,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512],
        "shape_code_encoder_hidden_dims": [512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512, 512, 512],
    },
    "bedroom_partial6d": {
        "procedure": "bedroom1",
        "num_epochs": 1001,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 250,
        "step_gamma": 0.5,
        "latent_size": 1024,
        "max_num_points": 5,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "shape_weight": 1000,
        "dimensions_matching_weight": 0.5,
        "encoder_hidden_dims": [512, 512, 512, 512],
        "decoder_hidden_dims": [512, 512, 512, 512],
        "shape_code_encoder_hidden_dims": [512, 512, 512, 512],
        "shape_code_encoder_output_size": 512,
        "shape_code_decoder_hidden_dims": [512, 512, 512, 512],
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
    # "lighting": "yellow",
}

colors_light = {
    "Bed": "#ADD8EE",
    "Cabinet": "#D3D3D3",
    "Nightstand": "#FFB580",
    "Chair": "#FF7F7F",
    "Pier/Stool": "#FF7F7F",
    "largeSofa": "#FFFF99",
    "largeTable": "#C89D7C",
    "tvStand": "#B0EEB0",
    "smallTable": "#b19cd9",
    # "lighting": "#FFFF99",
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
geometry_size = procedure_params["geometry_size"]
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