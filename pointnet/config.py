max_num_points = 5

geometry_size = 2 + 2 # position and dimensions
orientation_size = 0
num_classes = 2
code_size = 0
point_size = geometry_size + orientation_size + num_classes + code_size

geometric_weight = 1
categorical_weight = 0.03
existence_weight = 0.003
kld_loss_weight = 0