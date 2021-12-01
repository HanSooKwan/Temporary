import torch
import torch.nn as nn
from pathlib import Path

from modules.DAIN.MegaDepth.MegaDepth_model import HourGlass



# Getting Inverse Depth Map
class InvDepthMapEstimator(nn.Module):
    def __init__(self):
        super(InvDepthMapEstimator, self).__init__()
        self.depth_estimator = HourGlass()

    def load_pretrained(self, path=Path("models/MegaDepth/best_vanila_net_G.pth")):
        self.depth_estimator.load_state_dict(torch.load(path))

    def forward(self, image):
        # Get depth map / image must be normalized (0~1) image
        depth = self.depth_estimator(image)
        depth = torch.exp(depth[:, :1])
        return 1 / depth