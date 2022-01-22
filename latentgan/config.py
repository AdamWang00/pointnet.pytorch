model_name = "bedroom_latent_ori1"
iter_load = "200000"
print("latentgan", model_name, iter_load)

data_dir = "../data"

save_per_iters = 50000

model_generations_subdir = "Generations"

params_history = {
    "bedroom_latent1": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_partial6",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    },
    "bedroom_latent2": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_partial6e",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    },
    "bedroom_latent3": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_partial6h2",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    },
    "bedroom_latent4": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1c1",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    },
    "bedroom_latent4a": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1c1",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512, 512, 512]
    },
    "bedroom_latent4b": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full1c1",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512],
        "hidden_dims_d": [512, 512]
    },
    "bedroom_latent5": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full2b",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512, 512, 512]
    },
    "bedroom_latent5a": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_full2b",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    },
    "bedroom_latent_ori1": {
        "ae_model_class": "pointnetae",
        "ae_model_name": "bedroom_ori1",
        "ae_epoch_load": "latest",
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    }
}

params = params_history[model_name]

ae_model_class = params["ae_model_class"]
ae_model_name = params["ae_model_name"]
ae_epoch_load = params["ae_epoch_load"]

if ae_model_class == "pointnetvae":
    from pointnetvae.config import colors, colors_light, rooms_subdir, model_params_subdir, params_history as ae_params_history, procedure_params_all as ae_procedure_params_all
elif ae_model_class == "pointnetae":
    from pointnetae.config import colors, colors_light, rooms_subdir, model_params_subdir, params_history as ae_params_history, procedure_params_all as ae_procedure_params_all
else:
    raise Exception("ae model class not found:", ae_model_class)

ae_params = ae_params_history[ae_model_name]
ae_procedure = ae_params["procedure"]
ae_procedure_params = ae_procedure_params_all[ae_procedure]

# for loading LGAN model
generator_iters = params["generator_iters"]
batch_size = params["batch_size"]
learning_rate_g = params["learning_rate_g"]
learning_rate_d = params["learning_rate_d"]
z_dim = params["z_dim"]
hidden_dims_g = params["hidden_dims_g"]
hidden_dims_d = params["hidden_dims_d"]

# for loading AE model
max_num_points = ae_params["max_num_points"]
latent_size = ae_params["latent_size"]
encoder_hidden_dims = ae_params["encoder_hidden_dims"]
decoder_hidden_dims = ae_params["decoder_hidden_dims"]
shape_code_encoder_hidden_dims = ae_params["shape_code_encoder_hidden_dims"]
shape_code_encoder_output_size = ae_params["shape_code_encoder_output_size"]
shape_code_decoder_hidden_dims = ae_params["shape_code_decoder_hidden_dims"]

room_name = ae_procedure_params["room_name"]
geometry_size = ae_procedure_params["geometry_size"]
orientation_size = ae_procedure_params["orientation_size"]
num_categories = ae_procedure_params["num_categories"]
shape_size = ae_procedure_params["shape_size"]
point_size_intermediate = geometry_size + orientation_size + num_categories
point_size = point_size_intermediate + shape_size