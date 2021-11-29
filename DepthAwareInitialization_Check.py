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

from modules.DepthAwareFlowInitialization import DepthAwareFlowInitialization, DepthAwareFlowInitialization_Function_gradchecker
from modules.RAFT.core.raft import RAFT
from modules.RAFT.demo import load_image, viz_point
from modules.RAFT.core.utils.utils import InputPadder, InputPadder16
from modules.DAIN.MegaDepth.MegaDepth_model import HourGlass



###### Check if Defined Function works properly ######
device = "cuda"
if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

dafi_module = DepthAwareFlowInitialization(device=device)
dafi_module.to(device)
print(dafi_module.device)

# torch.manual_seed(0)

batch = 8
flow = torch.randn(batch,2,256,448,dtype=torch.float32,requires_grad=True)
depth = torch.randn(batch,256,448,dtype=torch.float32, requires_grad=True) + 2.0
flow = flow.to(device)
depth = depth.to(device)
output = dafi_module(flow, depth)
Loss = torch.sum(output)
Loss.backward()
# flow.grad
print(torch.sum(output[0]), torch.sum(output[1]), torch.sum(output[2]), torch.sum(output[3]))


###### Gradchecker ######
DepthAwareFlowInitialization_Function_gradchecker(device="cuda")




###### Now, let's see the depth-aware flow initialized output

# Make checkpoints directory
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')

# Arguments for pretrained RAFT and pretrained MegaDepth
args_RAFT = easydict.EasyDict({
    "small": False,
    "model": Path("models/" + "raft-sintel" + ".pth"),
    "dataset": 'chairs',
    "mixed_precision": True,
})
args_MEGADEPTH = easydict.EasyDict({
    "pretrained_path": Path("models/MegaDepth/best_vanila_net_G.pth")
})
if args_RAFT.small:
    args_RAFT.model = Path("models/raft-small.pth")


# Load Pretrained RAFT, MegaDepth Model
if torch.cuda.is_available():
    device = "cuda"
    model_RAFT = torch.nn.DataParallel(RAFT(args_RAFT))
    model_RAFT.load_state_dict(torch.load(args_RAFT.model))
    model_RAFT.to(device)
    model_RAFT.eval()
else:
    device = "cpu"
    model_RAFT = RAFT(args_RAFT)
    model_RAFT.load_state_dict(torch.load(args_RAFT.model))
    model_RAFT.to(device)
    model_RAFT.eval()


# Load Pretrained MEGADEPTH Model
if torch.cuda.is_available():
    device = "cuda"
    model_depth = torch.nn.DataParallel(HourGlass())
    model_depth.load_state_dict(torch.load(args_MEGADEPTH.pretrained_path))
    model_depth.to(device)
    model_depth.eval()
else:
    device = "cpu"
    model_depth = HourGlass()
    model_depth.load_state_dict(torch.load(args_MEGADEPTH.pretrained_path))
    model_depth.to(device)
    model_depth.eval()

### Set tensor type
device = "cuda"
if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)




for frame_num in range(16, 25):
    print("\n\n----------------------------------------------------------------------")
    prev = load_image(Path(
        "modules/RAFT/demo-frames/frame_00" + str(frame_num) + ".png"))  # load_image 함수: float32 형태의 0.0~255.0 사이 값들
    next = load_image(Path("modules/RAFT/demo-frames/frame_00" + str(frame_num + 1) + ".png"))
    print("Before inference of RAFT: prev shape / next shape: ", prev.shape, next.shape)
    b, c, height, width = prev.shape

    # RAFT
    padder = InputPadder16(prev.shape)  # 각변이 8로 나누어질 수 있도록 padding함 (양쪽으로 replication padding)
    prev_pad, next_pad = padder.pad(prev, next)
    # _, flow_forward = model_RAFT(prev_pad, next_pad, iters=25, test_mode=True)
    _, flow_backward = model_RAFT(next_pad, prev_pad, iters=25, test_mode=True)

    # Inverse Depth map
    # prev_pad = prev_pad.contiguous().float() / 255.0
    # prev_depth = model_depth(prev_pad)
    # prev_depth = torch.exp(prev_depth[:, 0])
    # prev_inv_depth = 1 / prev_depth
    next_pad = next_pad.contiguous().float() / 255.0
    next_depth = model_depth(next_pad)
    next_depth = torch.exp(next_depth[:, 0])
    next_inv_depth = 1 / next_depth

    # Depth-aware flow projection
    # daf_forward = DepthFlowProjectionModule_SuperNaive()(flow_forward, prev_inv_depth)
    DAFI = DepthAwareFlowInitialization(device=device)
    DAFI.to(device)
    daf_backward = DAFI(flow_backward, next_inv_depth)


    daf_backward = daf_backward.detach().cpu().numpy()
    next = next.detach().cpu().numpy()
    track_row = 280
    track_col = 870
    viz_point(next, daf_backward, track_row, track_col)