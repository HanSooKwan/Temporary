import argparse
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
import pefile
from modules.RAFT.core.raft import RAFT
from modules.RAFT.evaluate import validate_chairs, validate_sintel, validate_kitti
from modules.RAFT.demo import load_image, viz_point
from modules.RAFT.core.utils.utils import InputPadder, InputPadder16
from modules.DAIN import MegaDepth

# Make checkpoints directory
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')

# Arguments for pretrained RAFT and pretrained MegaDepth
# args_RAFT = easydict.EasyDict({
#     "small": False,
#     "model": Path("models/RAFT/" + "raft-sintel" + ".pth"),
#     "dataset": 'chairs',
#     "mixed_precision": True,
# })
args_MEGADEPTH = easydict.EasyDict({
    "pretrained_path": Path("models/MegaDepth/best_vanila_net_G.pth")
})
# if args_RAFT.small:
#     args_RAFT.model = Path("models/RAFT/raft-small.pth")


# Load Pretrained RAFT, MegaDepth Model
# if torch.cuda.is_available():
#     device = "cuda"
#     model_RAFT = torch.nn.DataParallel(RAFT(args_RAFT))
#     model_RAFT.load_state_dict(torch.load(args_RAFT.model))
#     model_RAFT.to(device)
#     model_RAFT.eval()
# else:
#     device = "cpu"
#     model_RAFT = RAFT(args_RAFT)
#     model_RAFT.load_state_dict(torch.load(args_RAFT.model))
#     model_RAFT.to(device)
#     model_RAFT.eval()


# Load Pretrained MEGADEPTH Model
if torch.cuda.is_available():
    device = "cuda"
    model_depth = torch.nn.DataParallel(MegaDepth.MegaDepth_model.HourGlass())
    model_depth.load_state_dict(torch.load(args_MEGADEPTH.pretrained_path))
    model_depth.to(device)
    model_depth.eval()
else:
    device = "cpu"
    model_depth = torch.nn.DataParallel(MegaDepth.MegaDepth_model.HourGlass())
    model_depth.load_state_dict(torch.load(args_MEGADEPTH.pretrained_path))
    model_depth.to(device)
    model_depth.eval()



# Saver
# save = h5py.File("GT_pf.hdf5", "w")


# Test Tracker
# track_row = 280
# track_col = 870
# print(f"Started at {track_row}, {track_col}")


print("\n\n----------------------------------------------------------------------")
prev = load_image(Path("modules/RAFT/demo-frames/frame_00" + str(frame_num) + ".png"))  # load_image 함수: float32 형태의 0.0~255.0 사이 값들

next = load_image(Path("modules/RAFT/demo-frames/frame_00" + str(frame_num + 1) + ".png"))
# print("Before inference of RAFT: prev shape / next shape: ", prev.shape, next.shape)
b, c, height, width = prev.shape

    # RAFT
padder = InputPadder16(prev.shape)  # 각변이 8로 나누어질 수 있도록 padding함 (양쪽으로 replication padding)
prev_pad, next_pad = padder.pad(prev, next)
    # _, flow_forward = model_RAFT(prev_pad, next_pad, iters=25, test_mode=True)
    # _, flow_backward = model_RAFT(next_pad, prev_pad, iters=25, test_mode=True)

# Inverse Depth map
prev_pad = prev_pad.contiguous().float() / 255.0
prev_depth = model_depth(prev_pad)
prev_depth = torch.exp(prev_depth[0, 0])
prev_inv_depth = 1 / prev_depth
    # next_pad = next_pad.contiguous().float() / 255.0
    # next_depth = model_depth(next_pad)
    # next_depth = torch.exp(next_depth[0, 0])
    # next_inv_depth = 1 / next_depth

    # Depth-aware flow projection
    # y_grid, x_grid = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
    # mesh = torch.stack([x_grid, y_grid], dim=0)
    # mesh0_1 = torch.round(mesh + 0.5 * flow_forward)





plt.figure()
plt.imshow(Ft_0)
plt.show()
    # Ft_1 = DepthFlowProjectionModule(next_pad.requires_grad)(flow_backward, next_depth)

    # flow_forward = flow_forward.detach().cpu().numpy()

# viz_point(prev, flow_up, track_row, track_col)

#     ################
#     image = load_image(
#         Path("modules/RAFT/demo-frames/frame_00" + str(
#             frame_num) + ".png"))  # load_image 함수: float32 형태의 0.0~255.0 사이 값들
#     image = image.contiguous().float() / 255.0  # Unlike RAFT (which normalizes uint8 images to [-1,1] INSIDE model), you need to normalize to [0,1] before forwarding
#
#     # Padding (Not required if dividable by 16)
#     padder = InputPadder16(image.shape)  # 각변이 16으로 나누어질 수 있도록 padding함 (양쪽으로 replication padding)
#     image_pad, image_pad = padder.pad(image, image)
#     log_depth = model_depth(image_pad)
#     # What we have got is log depth: exponential it
#     pred_depth = torch.exp(log_depth[0, 0])
#     # Reciprocals of depth
#     pred_inv_depth = 1 / pred_depth
#     # # Detach, turn to cpu, and change to numpy dtype
#     # pred_inv_depth = pred_inv_depth.detach().cpu().numpy()
#     # # Normalize using maximum value (just for visualization, do not use when training)
#     # pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)
#
#     ################
#
#
#
#
#     # Create dataset
#     flow_up = np.squeeze(flow_up)
#     save.create_dataset("flow_" + str(frame_num) + "->" + str(frame_num + 1) + "_row", data=flow_up[1])
#     save.create_dataset("flow_" + str(frame_num) + "->" + str(frame_num + 1) + "_column", data=flow_up[0])
#
#     print("flow estimation shape: ", flow_up.shape, "\nflow estimation dtype: ", flow_up.dtype,
#           "\ntracking point: row: ", track_row, "    col: ", track_col)
#     print("flow_up[0]", flow_up[0, track_row, track_col])
#     print("flow_up[1]", flow_up[1, track_row, track_col])
#     inc_col, inc_row = flow_up[:, track_row,
#                        track_col]  # row wise increment는 flow_up[1]에 해당 (즉, flow_up은 (x,y)에 대한 변화를 의미, NOT (row, col))
#     track_row = track_row + round(inc_row)
#     track_col = track_col + round(inc_col)
#
#     print("\nEstimated Next pixel of 300, 900: [", track_row, ",", track_col, "]")
# save.close()
#
# print("\n\n")
# print("Therefore, flow[1] is the y (height, row) direction optical flow estimation for all pixels, ")
# print("and flow[0] is the x (width, column) direction optical flow estimation for all pixels")
# print("row wise increment는 flow_up[1]에 해당 (즉, flow_up은 (x,y)에 대한 변화를 의미, NOT (row, col))")