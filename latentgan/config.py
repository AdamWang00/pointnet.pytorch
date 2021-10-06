model_name = "bedroom_latent_1d"
iter_load = "300000"

ae_epoch_load = "latest"

data_dir = "../data"

save_per_iters = 20000 # 20000

params_history = {
    "bedroom_latent_1a": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256],
        "hidden_dims_d": [256, 256]
    },
    "bedroom_latent_1b": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256],
        "hidden_dims_d": [256, 256, 256]
    },
    "bedroom_latent_1c": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256, 256],
        "hidden_dims_d": [256, 256, 256, 256]
    },
    "bedroom_latent_1d": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256, 256, 256],
        "hidden_dims_d": [256, 256, 256, 256, 256]
    },
    "bedroom_latent_2a": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full3",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256],
        "hidden_dims_d": [256, 256]
    },
    "bedroom_latent_2b": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full3",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256],
        "hidden_dims_d": [256, 256, 256]
    },
    "bedroom_latent_2c": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full3",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256, 256],
        "hidden_dims_d": [256, 256, 256, 256]
    },
    "bedroom_latent_2d": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full3",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256, 256, 256],
        "hidden_dims_d": [256, 256, 256, 256, 256]
    },
    "bedroom_latent_3a": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256],
        "hidden_dims_d": [256, 256, 256]
    },
    "bedroom_latent_3a2": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256],
        "hidden_dims_d": [256, 256, 256]
    },
    "bedroom_latent_3b": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256, 256],
        "hidden_dims_d": [256, 256, 256]
    },
    "bedroom_latent_3c": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256],
        "hidden_dims_d": [256, 256, 256, 256]
    },
    "bedroom_latent_3d": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [256, 256, 256, 256],
        "hidden_dims_d": [256, 256, 256, 256]
    },
    "bedroom_latent_3e": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1",
        "generator_iters": 300000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 64,
        "hidden_dims_g": [128, 256, 256],
        "hidden_dims_d": [256, 256, 256]
    },
}

params = params_history[model_name]

ae_model_class = params["ae_model_class"]
ae_model_name = params["ae_model_name"]

if ae_model_class == "pointnetvae":
    from pointnetvae.config import colors, colors_light, rooms_subdir, model_params_subdir, params_history as ae_params_history, procedure_params_all as ae_procedure_params_all
elif ae_model_class == "pointnetae":
    from pointnetae.config import colors, colors_light, rooms_subdir, model_params_subdir, params_history as ae_params_history, procedure_params_all as ae_procedure_params_all
else:
    raise Exception("ae model class not found:", ae_model_class)

ae_params = ae_params_history[ae_model_name]
ae_procedure = ae_params["procedure"]
ae_procedure_params = ae_procedure_params_all[ae_procedure]

max_num_points = ae_params["max_num_points"]
latent_size = ae_params["latent_size"]

generator_iters = params["generator_iters"]
batch_size = params["batch_size"]
learning_rate_g = params["learning_rate_g"]
learning_rate_d = params["learning_rate_d"]
z_dim = params["z_dim"]
hidden_dims_g = params["hidden_dims_g"]
hidden_dims_d = params["hidden_dims_d"]

room_name = ae_procedure_params["room_name"]
geometry_size = ae_procedure_params["geometry_size"]
orientation_size = ae_procedure_params["orientation_size"]
num_categories = ae_procedure_params["num_categories"]
code_size = ae_procedure_params["code_size"]
point_size = geometry_size + orientation_size + num_categories + code_size