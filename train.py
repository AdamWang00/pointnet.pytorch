import torch
import torch.optim as optim
import os

from pointnet.model import PointNetVAE
from pointnet.config import *
from utils import *


NUM_EXAMPLES = 1024
NUM_EPOCHS = 20
BATCH_SIZE = 64
LOAD_PATH = ""
SAVE_PATH = "experiments/model1"
LEARNING_RATE_INITIAL = 0.001


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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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

            cost_mat = get_cost_matrix_2d(reconstruction[:, 0:2], target[:, 0:2])
            cost_mat = cost_mat.detach().cpu()
            target_ind, matched_ind, unmatched_ind = get_assignment_problem_matchings(cost_mat)

            reconstruction_matched = reconstruction[matched_ind]
            # reconstruction_unmatched = reconstruction[unmatched_ind]
            target_existence = torch.zeros(max_num_points)
            target_existence[matched_ind] = 1
            target = target[target_ind]

            # print("============== SHAPES ===============")
            # print(reconstruction_matched[:, 0:geometry_size].shape,
            #     target[:, 0:geometry_size].shape,
            #     reconstruction_matched[:, geometry_size:geometry_size+num_classes].shape,
            #     target[:, geometry_size].shape,
            #     reconstruction[:, geometry_size+num_classes].shape,
            #     target_existence.shape,
            #     mu.shape,
            #     log_var.shape,
            # )

            losses[0] += geometric_weight * geometric_loss(
                reconstruction_matched[:, 0:geometry_size],
                target[:, 0:geometry_size]
            )
            losses[1] += categorical_weight * categorical_loss(
                reconstruction_matched[:, geometry_size:geometry_size+num_classes],
                target[:, geometry_size].long()
            )
            losses[2] += existence_weight * existence_loss(
                reconstruction[:, geometry_size+num_classes],
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