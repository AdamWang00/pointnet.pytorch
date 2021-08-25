model_name = "model2"
epoch_load = "latest"

params_history = {
    "model1": {
        "procedure": "table1",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.0,
        "dimensions_matching_weight": 0.5
    },
    "model2": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 64,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.1,
        "dimensions_matching_weight": 0.5
    },
    "model2a": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 64,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.03,
        "dimensions_matching_weight": 0.5
    },
    "model3": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.03,
        "dimensions_matching_weight": 0.5
    },
    "model4": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.1,
        "dimensions_matching_weight": 0.5
    },
    "model5": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.3,
        "dimensions_matching_weight": 0.5
    },
    "model6": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 1,
        "dimensions_matching_weight": 0.5
    },
    "model7": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 5,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 1,
        "dimensions_matching_weight": 0.5
    },
    "model8": {
        "procedure": "table2",
        "num_examples": 1024,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "step_size": 20,
        "step_gamma": 0.5,
        "latent_size": 4,
        "max_num_points": 20,
        "geometric_weight": 1,
        "categorical_weight": 1,
        "existence_weight": 0.003,
        "kld_loss_weight": 0.03,
        "dimensions_matching_weight": 0.5
    }
}

procedure_params_all = {
    "table1": {
        "geometry_size": 2 + 2, # position and dimensions
        "orientation_size": 0,
        "num_classes": 2,
        "code_size": 0,
    },
    "table2": {
        "geometry_size": 2 + 2, # position and dimensions
        "orientation_size": 0,
        "num_classes": 2,
        "code_size": 0,
    }
}

params = params_history[model_name]
procedure = params["procedure"]
procedure_params = procedure_params_all[procedure]

num_examples = params["num_examples"]
num_epochs = params["num_epochs"]
batch_size = params["batch_size"]
learning_rate = params["learning_rate"]
step_size = params["step_size"]
step_gamma = params["step_gamma"]

latent_size = params["latent_size"]
max_num_points = params["max_num_points"]

geometry_size = procedure_params["geometry_size"]
orientation_size = procedure_params["orientation_size"]
num_classes = procedure_params["num_classes"]
code_size = procedure_params["code_size"]
point_size = geometry_size + orientation_size + num_classes + code_size

geometric_weight = params["geometric_weight"]
categorical_weight = params["categorical_weight"]
existence_weight = params["existence_weight"]
kld_loss_weight = params["kld_loss_weight"]

dimensions_matching_weight = params["dimensions_matching_weight"]