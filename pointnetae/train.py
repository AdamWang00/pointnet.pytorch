import torch
import torch.optim as optim
import torch.utils.data as data_utils
import os
from pointnetae.model import PointNetAE
from pointnetae.config import *
from pointnetae.utils import *
from pointnetae.dataset import SceneDataset

# from torch.utils.data.dataloader import default_collate # for batching input scenes

REGRESS_UNMATCHED_DIM = True # regress dim of unmatched predictions to 0 (improves stability)

NUM_EPOCHS = num_epochs
BATCH_SIZE = batch_size
LOAD_PATH = ''
SAVE_PATH = os.path.join("experiments", model_name, model_params_subdir)
LEARNING_RATE_INITIAL = learning_rate
STEP_SIZE = step_size
STEP_GAMMA = step_gamma

base_dir = os.path.join(data_dir, room_name)
rooms_dir = os.path.join(base_dir, rooms_subdir)

model = PointNetAE()

if LOAD_PATH != '':
    model.load_state_dict(torch.load(LOAD_PATH))

if SAVE_PATH != '' and not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)
model = model.train().cuda()

scene_dataset = SceneDataset(rooms_dir, max_num_points, load_ram=True)

def collate_fn(batch):
    # return default_collate([t[0] for t in batch]), [t[1] for t in batch]
    return [t[0] for t in batch], [t[1] for t in batch]

scene_loader = data_utils.DataLoader(
    scene_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
    collate_fn=collate_fn
)

loss_log = []
geometric_loss_log = []
orientation_loss_log = []
categorical_loss_log = []
existence_loss_log = []
shape_loss_log = []

for epoch in range(NUM_EPOCHS):
    epoch_losses = [0, 0, 0, 0, 0] # geometric, orientation, categorical, existence, shape

    for i, scene_data in enumerate(scene_loader):
        scenes, targets = scene_data # scenes and targets are both lists of 2D tensors
        optimizer.zero_grad()
        
        losses = [0, 0, 0, 0, 0] # geometric, orientation, categorical, existence, shape
        for j in range(BATCH_SIZE):
            scene = scenes[j].transpose(1, 0).cuda() # need to transpose for Conv1d
            target = targets[j]
            cats = target[:, geometry_size + orientation_size].numpy().astype(int) # category indices
            target = target.cuda()

            # use single-element batches due to differently-shaped batch elements
            reconstruction_batch, latent_code_batch = model(scene.unsqueeze(0), np.expand_dims(cats, 0))
            reconstruction = reconstruction_batch[0]
            latent_code = latent_code_batch[0]

            cost_mat_position = get_cost_matrix_2d(reconstruction[:, 0:position_size], target[:, 0:position_size])
            cost_mat_dimension = get_cost_matrix_2d(reconstruction[:, position_size:position_size+dimension_size], target[:, position_size:position_size+dimension_size])
            cost_mat = cost_mat_position + dimensions_matching_weight * cost_mat_dimension
            cost_mat = cost_mat.detach().cpu()
            target_ind, matched_ind, unmatched_ind = get_assignment_problem_matchings(cost_mat)

            reconstruction_matched = reconstruction[matched_ind]
            reconstruction_unmatched = reconstruction[unmatched_ind]
            target_existence = torch.zeros(max_num_points)
            target_existence[matched_ind] = 1
            target = target[target_ind] # reorder target
            target_category_idx = target[:, geometry_size+orientation_size].long()

            # Geometry
            losses[0] += geometric_weight * geometric_loss(
                reconstruction_matched[:, 0:geometry_size],
                target[:, 0:geometry_size]
            )
            if REGRESS_UNMATCHED_DIM and reconstruction_unmatched.shape[0] > 0: # regress dimension of unmatched to zero
                losses[0] += geometric_weight * geometric_loss(
                    reconstruction_unmatched[:, position_size:position_size+dimension_size],
                    torch.zeros_like(reconstruction_unmatched[:, position_size:position_size+dimension_size])
                )
            # Orientation
            losses[1] += orientation_weight * orientation_loss(
                reconstruction_matched[:, geometry_size:geometry_size+orientation_size],
                target[:, geometry_size:geometry_size+orientation_size]
            )
            # Category
            losses[2] += categorical_weight * categorical_loss(
                reconstruction_matched[:, geometry_size+orientation_size:geometry_size+orientation_size+num_categories],
                target_category_idx
            )
            # Existence
            losses[3] += existence_weight * existence_loss(
                reconstruction[:, geometry_size+orientation_size+num_categories],
                target_existence.cuda()
            )
            # Shape
            shape_codes = torch.zeros(target.shape[0], shape_size).cuda()
            for k in range(target.shape[0]):
                x = torch.cat(
                    (
                        latent_code,
                        reconstruction_matched[k, 0:geometry_size+orientation_size]
                    )
                )
                shape_codes[k, :] = model.decode_shape(x, target_category_idx[k])
            losses[4] += shape_weight * shape_loss(
                shape_codes,
                target[:, geometry_size+orientation_size+1:]
            )

        loss = 0
        for li in range(len(losses)):
            loss += losses[li]
            epoch_losses[li] += losses[li].item()

        # if opt.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

        print('[%d: %d] train loss: %f (%f, %f, %f, %f, %f)' % (
            epoch + 1, i + 1, loss.item(), losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item(), losses[4].item()
        ))

    epoch_loss = 0
    for li in range(len(epoch_losses)):
        epoch_loss += epoch_losses[li]

    print('EPOCH %d train loss: %f (%f, %f, %f, %f, %f)' % (
        epoch + 1, epoch_loss, epoch_losses[0], epoch_losses[1], epoch_losses[2], epoch_losses[3], epoch_losses[4]
    ))

    loss_log.append(epoch_loss)
    geometric_loss_log.append(epoch_losses[0])
    orientation_loss_log.append(epoch_losses[1])
    categorical_loss_log.append(epoch_losses[2])
    existence_loss_log.append(epoch_losses[3])
    shape_loss_log.append(epoch_losses[4])

    scheduler.step()

    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), '%s/%d.pth' % (SAVE_PATH, epoch + 1))

torch.save(
    {
        "loss": loss_log,
        "geometric_loss": geometric_loss_log,
        "orientation_loss": orientation_loss_log,
        "categorical_loss": categorical_loss_log,
        "existence_loss": existence_loss_log,
        "shape_loss": shape_loss_log
    },
    os.path.join("experiments", model_name, "Logs.pth")
)

torch.save(model.state_dict(), '%s/latest.pth' % (SAVE_PATH))