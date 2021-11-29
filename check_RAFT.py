import argparse
import pathlib

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import easydict
from pathlib import Path
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image

from modules.RAFT.core.raft import RAFT
from modules.RAFT.demo import load_image, viz_point
from modules.RAFT.core.utils.utils import InputPadder
from modules.DAIN import MegaDepth

args_RAFT = easydict.EasyDict({
    "small": False,
    "model": Path("models/raft-sintel.pth"),
    "dataset": 'sintel',
    "mixed_precision": True,
})

if args_RAFT.small:
    args_RAFT.model = Path(r"D:\Downloads\GETTHIS\models\RAFT\raft-small.pth")

# if not os.path.isdir(Path('models/checkpoints')):
#     os.mkdir(Path('models/checkpoints'))


if torch.cuda.is_available():
    device = "cuda"
    model = torch.nn.DataParallel(RAFT(args_RAFT))
    model.load_state_dict(torch.load(args_RAFT.model))
    model.to(device)
    model.eval()
else:
    device = "cpu"
    model = RAFT(args_RAFT)
    model.load_state_dict(torch.load(args_RAFT.model))
    model.to(device)
    model.eval()

# Saver
# save = h5py.File("GT_pf.hdf5", "w")

track_row = 280
track_col = 870
print(f"Started at {track_row}, {track_col}")
for frame_num in range(16, 25):
    print("\n\n----------------------------------------------------------------------")
    prev = load_image(Path("modules/RAFT/demo-frames/frame_00" + str(frame_num) + ".png"))
    next = load_image(Path("modules/RAFT/demo-frames/frame_00" + str(frame_num + 1) + ".png"))
    print("Before inference of RAFT: prev shape / next shape: ", prev.shape, next.shape)
    b, c, height, width = prev.shape
    # RAFT
    padder = InputPadder(prev.shape)  # 각변이 8로 나누어질 수 있도록 padding함 (양쪽으로 replication padding)
    prev_pad, next_pad = padder.pad(prev, next)
    _, flow_up = model(prev_pad, next_pad, iters=25, test_mode=True)
    prev = prev.detach().cpu().numpy()
    flow_up = flow_up.detach().cpu().numpy()
    viz_point(prev, flow_up, track_row, track_col)

    # Create dataset
    flow_up = np.squeeze(flow_up)
    # save.create_dataset("flow_" + str(frame_num) + "->" + str(frame_num + 1) + "_row", data=flow_up[1])
    # save.create_dataset("flow_" + str(frame_num) + "->" + str(frame_num + 1) + "_column", data=flow_up[0])

    print("flow estimation shape: ", flow_up.shape, "\nflow estimation dtype: ", flow_up.dtype,
          "\ntracking point: row: ", track_row, "    col: ", track_col)
    print("flow_up[0]", flow_up[0, track_row, track_col])
    print("flow_up[1]", flow_up[1, track_row, track_col])
    inc_col, inc_row = flow_up[:, track_row,
                       track_col]  # row wise increment는 flow_up[1]에 해당 (즉, flow_up은 (x,y)에 대한 변화를 의미, NOT (row, col))
    track_row = track_row + round(inc_row)
    track_col = track_col + round(inc_col)

    print("\nEstimated Next pixel: [", track_row, ",", track_col, "]")
# save.close()

print("\n\n")
print("Therefore, flow[1] is the y (height, row) direction optical flow estimation for all pixels, ")
print("and flow[0] is the x (width, column) direction optical flow estimation for all pixels")
print("row wise increment는 flow_up[1]에 해당 (즉, flow_up은 (x,y)에 대한 변화를 의미, NOT (row, col))")