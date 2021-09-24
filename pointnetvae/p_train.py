import torch
import torch.optim as optim
import os
from pointnetvae.model import PointNetVAE
from pointnetvae.config import *
from pointnetvae.utils import *


NUM_EXAMPLES = num_examples
NUM_EPOCHS = num_epochs
BATCH_SIZE = batch_size
LOAD_PATH = ''
SAVE_PATH = os.path.join("experiments", model_name, model_params_subdir)
LEARNING_RATE_INITIAL = learning_rate
STEP_SIZE = step_size
STEP_GAMMA = step_gamma


num_batches = NUM_EXAMPLES // BATCH_SIZE
data = []
for i in range(num_batches):
    data.append(generate_scene(BATCH_SIZE))

model = PointNetVAE()

if LOAD_PATH != '':
    model.load_state_dict(torch.load(LOAD_PATH))

if SAVE_PATH != '' and not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)
model = model.train()
model.cuda()

for epoch in range(NUM_EPOCHS):
    epoch_losses = [0, 0, 0, 0] # geometric, categorical, existence, kld

    for i in range(num_batches):
        optimizer.zero_grad()

        scene, target_list = data[i]
        scene = scene.transpose(2, 1)
        scene = scene.cuda()

        # forward
        reconstruction_batch, _, trans_feat, mu, log_var = model(scene)
        
        losses = [0, 0, 0, 0] # geometric, categorical, existence, kld
        for j in range(BATCH_SIZE):
            reconstruction = reconstruction_batch[j]
            target = target_list[j]
            target = target.cuda()

            cost_mat_position = get_cost_matrix_2d(reconstruction[:, 0:2], target[:, 0:2])
            cost_mat_dimension = get_cost_matrix_2d(reconstruction[:, 2:4], target[:, 2:4])
            cost_mat = cost_mat_position + dimensions_matching_weight * cost_mat_dimension
            cost_mat = cost_mat.detach().cpu()
            target_ind, matched_ind, unmatched_ind = get_assignment_problem_matchings(cost_mat)

            reconstruction_matched = reconstruction[matched_ind]
            reconstruction_unmatched = reconstruction[unmatched_ind]
            target_existence = torch.zeros(max_num_points)
            target_existence[matched_ind] = 1
            target = target[target_ind]

            # Geometry
            losses[0] += geometric_weight * geometric_loss(
                reconstruction_matched[:, 0:geometry_size],
                target[:, 0:geometry_size]
            )
            if reconstruction_unmatched.shape[0] > 0:
                losses[0] += geometric_weight * geometric_loss(
                    reconstruction_unmatched[:, 2:4], # regress only dimension of unmatched
                    torch.zeros_like(reconstruction_unmatched[:, 2:4])
                )
            # Category
            losses[1] += categorical_weight * categorical_loss(
                reconstruction_matched[:, geometry_size:geometry_size+num_categories],
                target[:, geometry_size].long()
            )
            # Existence
            losses[2] += existence_weight * existence_loss(
                reconstruction[:, geometry_size+num_categories],
                target_existence.cuda()
            )

        losses[3] = kld_loss_weight * kld_loss(mu, log_var)

        loss = 0
        for li in range(len(losses)):
            loss += losses[li]
            epoch_losses[li] += losses[li].item()

        # if opt.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

        print('[%d: %d/%d] train loss: %f (%f, %f, %f, %f)' % (
            epoch + 1, i + 1, num_batches, loss.item(), losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()
        ))

    epoch_loss = 0
    for li in range(len(epoch_losses)):
        epoch_loss += epoch_losses[li]

    print('EPOCH %d train loss: %f (%f, %f, %f, %f)' % (
        epoch + 1, epoch_loss, epoch_losses[0], epoch_losses[1], epoch_losses[2], epoch_losses[3]
    ))

    scheduler.step()

    torch.save(model.state_dict(), '%s/%d.pth' % (SAVE_PATH, epoch))

torch.save(model.state_dict(), '%s/latest.pth' % (SAVE_PATH))