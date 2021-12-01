import torch
import torch.nn as nn




class RelativeInvDepth(nn.Module): # Normalize inv_depth, so it becomes relative value
    def __init__(self):
        super(RelativeInvDepth, self).__init__()
    def forward(self, inv_depth):
        # Normalize to mean 0, std 1

        B, Ch, H, W = inv_depth.shape # B, 1, H, W
        inv_depth = inv_depth.view(B, Ch, H * W)
        mean = inv_depth.mean(dim=2).view(B, Ch, 1, 1)
        # print(mean.shape)
        std = inv_depth.std(dim=2).view(B, Ch, 1, 1)
        inv_depth = inv_depth.view(B, Ch, H, W)
        return (inv_depth - mean) / std

class RelativeContext(nn.Module): # Normalize context so high value means contextual importance
                                  # Unlike inv_depth, you must normalize along channel dimension too
    def __init__(self):
        super(RelativeContext, self).__init__()
    def forward(self, context):
        # Normalize to mean 0, std 1
        B, Ch, H, W = context.shape # B, 1, H, W
        context = context.view(B, Ch * H * W)
        mean = context.mean(dim=1).view(B, 1, 1, 1)
        # print(mean.shape)
        std = context.std(dim=1).view(B, 1, 1, 1)
        context = context.view(B, Ch, H, W)
        return (context - mean) / std

class DepthAndContextAware(nn.Module):
    def __init__(self, in_chans, hidden_ch=64, input_relative_depth_context = True):
        super(DepthAndContextAware, self).__init__()
        self.input_relative_depth_context = input_relative_depth_context
        if not self.keep_depth_and_context:
            self.relative_depth = RelativeInvDepth()
            self.relative_context = RelativeContext()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=hidden_ch, kernel_size=7, stride=1, padding=1),
            # Used kernel size 7 to apply the large spatial resoultion at first
            nn.ReLU6() # Used to clip too high values that disrupt the flow too badly
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_ch, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, flow, inv_depth, context):
        # Relative depth map, Relative context map
        if not self.input_relative_depth_context:
            inv_depth = self.relative_depth(inv_depth)
            context = self.relative_context(context)

        # Go through model
        output_flow = self.layer3(self.layer2(self.layer1(torch.cat([flow, inv_depth, context], dim=1))))

        # Residual connection
        output_flow = output_flow + flow
        return output_flow, inv_depth, context

class FlowGeneration(nn.Module):
    def __init__(self, context_channels = 51, num_iter = 3):
        super(FlowGeneration,self).__init__()

        self.context_channels = context_channels
        self.num_iteration = num_iter

        self.layers = nn.Sequential(
            [DepthAndContextAware(in_chans=context_channels+3,input_relative_depth_context=False)]
            + [DepthAndContextAware(in_chans=context_channels+3,input_relative_depth_context=True)] * (num_iter - 1)
        )

    def forward(self, flow, inv_depth, context):
        # depth and context normalization: DONE IN LAYERS
        flow, _, _ = self.layers(flow, inv_depth, context)

        return flow