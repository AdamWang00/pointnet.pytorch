model_name = "bedroom_latent_1"
epoch_load = "latest"

ae_epoch_load = "latest"

data_dir = "../data"

params_history = {
    "bedroom_latent_1": {
        "ae_model_class": "pointnetvae",
        "ae_model_name": "bedroom_full1",
        "num_epochs": 1 * 5,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
    }
}

params = params_history[model_name]

ae_model_class = params["ae_model_class"]
ae_model_name = params["ae_model_name"]

if ae_model_class == "pointnetvae":
    from pointnetvae.config import colors, colors_light, rooms_subdir, model_params_subdir, params_history as ae_params_history, procedure_params_all as ae_procedure_params_all
else:
    raise Exception("ae model class not found:", ae_model_class)

ae_params = ae_params_history[ae_model_name]
ae_procedure = ae_params["procedure"]
ae_procedure_params = ae_procedure_params_all[ae_procedure]

max_num_points = ae_params["max_num_points"]

num_epochs = params["num_epochs"]
batch_size = params["batch_size"]
learning_rate_g = params["learning_rate_g"]
learning_rate_d = params["learning_rate_d"]

room_name = ae_procedure_params["room_name"]
geometry_size = ae_procedure_params["geometry_size"]
orientation_size = ae_procedure_params["orientation_size"]
num_categories = ae_procedure_params["num_categories"]
code_size = ae_procedure_params["code_size"]
point_size = geometry_size + orientation_size + num_categories + code_size