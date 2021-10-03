model_name = "bedroom_full1"
epoch_load = "latest"

data_dir = "../data"
rooms_subdir = "Rooms"

model_params_subdir = "ModelParameters"

params_history = {
    "bedroom_full1": {
        "procedure": "bedroom2",
        "num_epochs": 800,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 200,
        "step_gamma": 0.5,
        "latent_size": 256,
        "max_num_points": 20,
        "geometric_weight": 1,
        "orientation_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 1,
        "dimensions_matching_weight": 0.5
    },
}

procedure_params_all = {
    "bedroom2": {
        "room_name": "Bedroom2",
        "geometry_size": 2 + 2, # position and dimensions
        "orientation_size": 2,
        "num_categories": 10,
        "code_size": 0,
    },
}

colors = {
    "Bed": "blue",
    "Cabinet/Shelf/Desk": "grey",
    "Chair": "red",
    "Pier/Stool": "red",
    "Sofa": "yellow",
    "Table": "green",
    "Nightstand": "orange",
    "TV Stand": "green",
    "Coffee Table": "green",
    "Wardrobe": "purple",
    # "Pendant Lamp": "yellow",
    # "Ceiling Lamp": "yellow",
}

colors_light = {
    "Bed": "#ADD8EE",
    "Nightstand": "#FFB580",
    "Cabinet/Shelf/Desk": "#D3D3D3",
    "Chair": "#FF7F7F",
    "Pier/Stool": "#FF7F7F",
    "Sofa": "#FFFF99",
    "Table": "#B0EEB0",
    "TV Stand": "#B0EEB0",
    "Coffee Table": "#B0EEB0",
    "Wardrobe": "#b19cd9",
    # "Pendant Lamp": "#FFFF99",
    # "Ceiling Lamp": "#FFFF99",
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
code_size = procedure_params["code_size"]
point_size = geometry_size + orientation_size + num_categories + code_size

geometric_weight = params["geometric_weight"]
orientation_weight = params["orientation_weight"]
categorical_weight = params["categorical_weight"]
existence_weight = params["existence_weight"]

dimensions_matching_weight = params["dimensions_matching_weight"]