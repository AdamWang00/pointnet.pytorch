import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from pointnetae.config import *

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

logs = torch.load(os.path.join("experiments", model_name, "Logs.pth"))

num_epochs = len(logs["loss"])

smoothed_loss_41 = running_mean(logs["loss"], 41)

fig, ax = plt.subplots()

ax.plot(
    np.arange(num_epochs),
    logs["loss"],
    "#82c6eb",
    np.arange(num_epochs),
    logs["geometric_loss"],
    "#52eb6e",
    np.arange(num_epochs),
    logs["orientation_loss"],
    "#eb5252",
    np.arange(num_epochs),
    logs["categorical_loss"],
    "#ffaf59",
    np.arange(num_epochs),
    logs["existence_loss"],
    "#ebe652",
    np.arange(num_epochs),
    logs["shape_loss"],
    "#6759ff",
    # smoothed_loss_41,
    # "#2a9edd"
)

ax.grid()
plt.show()