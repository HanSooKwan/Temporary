import torch
import torch.nn as nn
import easydict
from pathlib import Path

from modules.AdaptiveWarpingLayer import AdaptiveWarpingLayer
from modules.FlowGeneration import FlowGeneration
from modules.FrameSynthesis import FrameSynthesis
from modules.DepthAwareFlowInitialization import DepthAwareFlowInitialization
from modules.ContextExtraction import ContextExtraction
from modules.InvDepthExtraction import InvDepthMapEstimator
from modules.KernelEstimation import KernelEstimation


# Cell of LSTM-like structure
class NextImagePrediction(nn.Module):
    def __init__(self, device="cuda", inplanes=16, use_pretrained_megadepth=True):
        super(NextImagePrediction, self).__init__()

        # Get inv_depth map of image
        self.inv_depth_estimator = InvDepthMapEstimator()
        if use_pretrained_megadepth:
            self.inv_depth_estimator.initialize(path=Path("models/MegaDepth/best_vanila_net_G.pth"))

        # Get context map of image --> context channels would be 3 * inplanes
        self.context_extractor = ContextExtraction(num_blocks=3, inplanes=inplanes) # You can change to 32

        # Predict next flow
        self.depth_aware_initialization = DepthAwareFlowInitialization(device=device, backward=True)
        self.flow_generation = FlowGeneration(context_channels=2+1+3*inplanes, num_iter=3)

        # Kernel estimation from image
        self.kernel_estimator = KernelEstimation(in_chans=3, out_chans=16, leaky_slope=0.02)

        # Adaptive Warping Layer
        self.adaptive_warping_layer = AdaptiveWarpingLayer(device=device, backward=True)

        # Frame synthesizer
        self.frame_synthesizer = FrameSynthesis(in_chans=3+1+3*inplanes,out_chans=3)

    def load_pretrained_MegaDepth(self):
        self.inv_depth_estimator.load_pretrained(path=Path("models/MegaDepth/best_vanila_net_G.pth"))

    def forward(self, image, flow_before): # Please input normalized (0~1) float32 type image, NOT uint8 type image
        # Extract inv_depth_map and context from I(t)
        inv_depth = self.inv_depth_estimator(image)
        context = self.context_extractor(image)

        # Depth aware flow initialization, and then flow generation
        flow_after = self.flow_generation(self.depth_aware_initialization(flow_before, inv_depth), inv_depth, context)

        # Kernel estimation
        kernel = self.kernel_estimator(image)

        # Concat image, inv depth map, and context map wrt channel dimension
        image = torch.cat([image, inv_depth, context], dim=1)

        # Warp image, inv depth map, and context map
        warped = self.adaptive_warping_layer(image, kernel, flow_after)

        # Synthesize next frame (Adds image inside frame_synthesizer using input[:,:3]
        next_image = self.frame_synthesizer(warped)

        return next_image, flow_after