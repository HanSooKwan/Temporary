import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class AdaptiveWarpingLayer(nn.Module):
    def __init__(self, device, backward=True):
        super(AdaptiveWarpingLayer, self).__init__()
        self.device = device
        self.backward = backward  # If backward is False: use I(t-1) -> I(t) to initialize I(t) -> I(t+1)
        # If backward is True: use I(t) -> I(t-1) to initialize I(t+1) -> I(t)

    def forward(self, image, kernel, flow):
        return AdaptiveWarpingLayer_Function.apply(image, kernel, flow, flow.requires_grad, self.backward, self.device)


class AdaptiveWarpingLayer_Function(Function):
    def __init__(self):
        super(AdaptiveWarpingLayer_Function, self).__init__()

    @staticmethod
    def forward(ctx, image, kernel, flow, requires_grad, is_backward, device):
        ## Recall: If is_backward is False, flow is flow from I(t) -> I(t+1)
        ## Recall: If is_backward is True, flow is flow from I(t+1) -> I(t)

        ## Recall: image can be Context (containing image): 195 channels / Depth: 1 channel
        ## Recall: kernel shape is B*16*H*W
        ## Recall: flow shape is B*2*H*W

        # Get shape
        B, Ch, H, W = image.shape
        # _, Ch_depth, _, _ = depth.shape

        # Define meshgrid
        y_grid, x_grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        batch_mesh = torch.arange(start=0, end=B).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(
            dim=-1)  # Shape: B, 1, 1, 1
        channel_mesh = torch.arange(start=0, end=Ch).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(
            dim=-1)  # Shape: 1, Ch, 1, 1
        mesh_of_tp1 = torch.stack([x_grid, y_grid], dim=0).unsqueeze(dim=0)

        # For I(t+1) image, where would the corresponding pixels in I(t) be?? (Estimate using flow from I(t+1)->I(t) IF is_backward is True, ELSE use I(t)->I(t+1) with missing holes) [Getting x+[f(x)]]

        if is_backward:  # Use backward flow (I(t+1)->I(t))
            if device == "cuda":
                r = torch.Tensor([[-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                                  [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2]]).type(
                    torch.cuda.LongTensor).T.unsqueeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                mesh_of_t = torch.floor(mesh_of_tp1 + flow).type(torch.cuda.LongTensor)  # Shape: B, 2, H, W
                theta = flow - torch.floor(flow)  # Fractional part, Shape: B, 2, H, W
                mesh_of_t_16 = mesh_of_t.repeat(16, 1, 1, 1, 1) + r  # Shape: 16, B, 2, H, W

            elif device == "cpu":
                r = torch.Tensor([[-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                                  [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2]]).type(
                    torch.LongTensor).T.unsqueeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                mesh_of_t = torch.floor(mesh_of_tp1 + flow).type(torch.LongTensor)  # Shape: B, 2, H, W
                theta = flow - torch.floor(flow)  # Fractional part, Shape: B, 2, H, W
                mesh_of_t_16 = mesh_of_t.repeat(16, 1, 1, 1, 1) + r  # Shape: 16, B, 2, H, W

        # else: # Use forward flow (I(t))->I(t+1))
        #     if device == "cuda":
        #         mesh_of_tp1 = torch.floor(mesh_of_tp1 - flow).type(torch.cuda.LongTensor) # Shape: B, 2, H, W
        #     elif device == "cpu":
        #         mesh_of_tp1 = torch.floor(mesh_of_tp1 - flow).type(torch.LongTensor)

        # Boolean array checking if in_range (shape: 4, B, H, W)
        in_range = torch.stack([mesh_of_t_16[:, :, 0] >= 0, mesh_of_t_16[:, :, 0] < W, mesh_of_t_16[:, :, 1] >= 0,
                                mesh_of_t_16[:, :, 1] < H], dim=0)  # Shape: 4, 16, B, H, W
        in_range = torch.all(in_range, dim=0)  # Shape: 16, B, H, W

        # Only consider "future is in-range" pixels
        mesh_of_t_16 = in_range.unsqueeze(dim=2) * mesh_of_t_16  # Shape: 16, B, 2, H, W

        # Indices (y, x)
        y_of_t_16 = mesh_of_t_16[:, :, 1].unsqueeze(dim=2)
        x_of_t_16 = mesh_of_t_16[:, :, 0].unsqueeze(dim=2)

        # Correspondence in I(t) of pixels in I(t+1) (for image, depth, context)
        image_picked_16 = image[batch_mesh.unsqueeze(dim=0), channel_mesh.unsqueeze(dim=0), y_of_t_16, x_of_t_16]
        image_picked_16 = in_range.unsqueeze(dim=2) * image_picked_16  # Set image pixel values to 0 if out of range
        # depth_picked_16 = depth[batch_mesh.unsqueeze(dim=0),channel_mesh_depth.unsqueeze(dim=0),y_of_t_16, x_of_t_16]
        # depth_picked_16 = in_range.unsqueeze(dim=2) * depth_picked_16

        # Kernel
        kernel = kernel.permute(1, 0, 2, 3).unsqueeze(dim=2)  # Shape: 16, B, 1, H, W

        # Bilinear interpolation coefficients
        theta_u = theta[:, 0].unsqueeze(dim=1)  # Shape: B,1,H,W
        theta_v = theta[:, 1].unsqueeze(dim=1)  # Shape: B,1,H,W
        theta_16 = torch.stack([1 - theta_u] * 8 + [theta_u] * 8, dim=0) * torch.stack(
            [1 - theta_v, 1 - theta_v, theta_v, theta_v] * 4, dim=0)  # Shape: 16, B, 1, H, W

        # Multiply all
        image_warped_16 = image_picked_16 * kernel * theta_16  # shape: 16, B, Ch, H, W

        # Sum all for dim=0 (Sum for 16 corresponding pixels)
        image_warped = torch.sum(image_warped_16, dim=0)

        # Save tensors for backward
        ctx.save_for_backward(image_picked_16, mesh_of_t_16, in_range, kernel, theta_16, theta_u, theta_v)

        return image_warped

    @staticmethod
    def backward(ctx, grad_image_warped):  # shape: B, Ch, H, W

        # Declare gradients
        grad_image = grad_kernel = grad_flow = None

        # Retrieve saved tensors from forward
        image_picked_16, mesh_of_t_16, in_range, kernel, theta_16, theta_u, theta_v = ctx.saved_tensors
        # RECALL: image_picked_16 --> Image values of In-range 16 correspondences in I(t) of pixel in I(t+1) / Shape: 16, B, Ch, H, W
        # RECALL: mesh_of_t_16--> Coordinates of In-range 16 correspondences in I(t) of pixel in I(t+1) / Shape: 16, B, 2, H, W
        # RECALL: in_range--> Boolean tensor of checking if 16 correspondences (of that pixel) are in-range of image / Shape: 16, B, H, W
        # RECALL: kernel --> 16 interpolation coefficients (of that pixel) / # Shape: 16, B, 1, H, W
        # RECALL: theta_16 --> 16 Bilinear interpolation coefficients of correspondence in I(t) / Shape: 16, B, 1, H, W
        # RECALL: theta_u, theta_v --> Fractional parts of flow from I(t+1)->I(t) / Shape: B, 1, H, W

        # Get shape
        _, B, Ch, H, W = image_picked_16.shape

        if ctx.needs_input_grad[0]:
            # Ravel (Warning: DO NOT ADD torch.arange(0,16) since we are going to add it in first place, NOT sum it by dimension after unraveling)
            unravel_length = 16 * B * H * W
            if B == 1:
                if Ch == 1:
                    mesh_t_16_raveled = torch.reshape(mesh_of_t_16[:, :, 0]
                                                      + mesh_of_t_16[:, :, 1] * W
                                                      #   + torch.arange(start=0,end=16).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) * B * H * W
                                                      , [unravel_length])  # Change to 1D

                if Ch >= 2:
                    mesh_t_16_raveled = torch.reshape(mesh_of_t_16[:, :, 0].unsqueeze(dim=0)
                                                      + mesh_of_t_16[:, :, 1].unsqueeze(dim=0) * W
                                                      + torch.arange(start=0, end=Ch).unsqueeze(dim=-1).unsqueeze(
                        dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) * H * W
                                                      #   + torch.arange(start=0,end=16).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) * B * H * W
                                                      , [Ch * unravel_length])  # Change to 1D
            else:
                if Ch == 1:
                    mesh_t_16_raveled = torch.reshape(mesh_of_t_16[:, :, 0]
                                                      + mesh_of_t_16[:, :, 1] * W
                                                      + torch.arange(start=0, end=B).unsqueeze(dim=0).unsqueeze(
                        dim=-1).unsqueeze(dim=-1) * H * W
                                                      #   + torch.arange(start=0,end=16).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) * B * H * W
                                                      , [unravel_length])  # Change to 1D
                if Ch >= 2:
                    mesh_t_16_raveled = torch.reshape(mesh_of_t_16[:, :, 0]
                                                      + mesh_of_t_16[:, :, 1] * W
                                                      + torch.arange(start=0, end=B).unsqueeze(dim=0).unsqueeze(
                        dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) * H * W
                                                      + torch.arange(start=0, end=Ch).unsqueeze(dim=-1).unsqueeze(
                        dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) * B * H * W
                                                      #   + torch.arange(start=0,end=16).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) * B * H * W
                                                      , [Ch * unravel_length])  # Change to 1D
            # Note: mesh_t_16_raveled goes to index

            # Ravel these stuff
            # in_range_raveled = torch.reshape(in_range, [unravel_length]).unsqueeze(dim=0) # To make weights 0 if z=x+[f(x)]+r is out of range
            # kernel_16_raveled = torch.reshape(kernel, [unravel_length]).unsqueeze(dim=0)
            # theta_16_raveled = torch.reshape(theta_16, [unravel_length]).unsqueeze(dim=0)
            in_range_raveled = torch.reshape(in_range,
                                             [unravel_length])  # To make weights 0 if z=x+[f(x)]+r is out of range
            kernel_16_raveled = torch.reshape(kernel, [unravel_length])
            theta_16_raveled = torch.reshape(theta_16, [unravel_length])

            if Ch >= 2:
                in_range_raveled = torch.cat([in_range_raveled] * Ch)
                kernel_16_raveled = torch.cat([kernel_16_raveled] * Ch)
                theta_16_raveled = torch.cat([theta_16_raveled] * Ch)

            # Ravel gradient of warped image
            grad_image_warped_raveled = torch.reshape(
                torch.permute(torch.stack([grad_image_warped] * 16, dim=0), [2, 0, 1, 3, 4]), [Ch * unravel_length])

            # To calculate gradient wrt I^t(z), we must multiply raveled tensors
            mult_raveled = grad_image_warped_raveled * theta_16_raveled * kernel_16_raveled * in_range_raveled

            # Now, accumulate to calculate gradient wrt I^t(z)
            minlength = Ch * B * H * W
            grad_image = torch.bincount(input=mesh_t_16_raveled, weights=mult_raveled, minlength=minlength)[
                         :minlength]  # Length: Ch*B*H*W
            grad_image = torch.permute(torch.reshape(grad_image, [Ch, B, H, W]), [1, 0, 2, 3])  # Reshaping process

            # if Ch == 3: # image
            #     grad_image_R = torch.bincount(input=mesh_t_16_raveled, weights=mult_raveled[0], minlength=minlength)[:minlength]
            #     grad_image_G = torch.bincount(input=mesh_t_16_raveled, weights=mult_raveled[1], minlength=minlength)[:minlength] # Set minlength as B*H*W, because we accumulate 16 values at this point
            #     grad_image_B = torch.bincount(input=mesh_t_16_raveled, weights=mult_raveled[2], minlength=minlength)[:minlength] # Set minlength as B*H*W, because we accumulate 16 values at this point

            #     # Stack and reshape to get grad_image
            #     grad_image = torch.permute(torch.reshape(torch.stack([grad_image_R, grad_image_G, grad_image_B], dim=0), [Ch,B,H,W]),[1,0,2,3])

            # if Ch == 1: # Depth
            #     grad_image = torch.bincount(input=mesh_t_16_raveled, weights=mult_raveled[0], minlength=minlength)[:minlength]
            #     grad_image = torch.reshape(grad_image, [B,1,H,W])

        if ctx.needs_input_grad[1]:
            grad_kernel = torch.sum(image_picked_16 * theta_16 * grad_image_warped.unsqueeze(dim=0),
                                    dim=2)  # Shape: 16, B, H, W
            grad_kernel = grad_kernel.permute(1, 0, 2, 3)  # Reshape for output

        if ctx.needs_input_grad[2]:
            # Get gradient of image pixel value wrt flow: u-direction
            grad_bilinearcoef_flow_u = torch.stack([-1 + theta_v, -1 + theta_v, -theta_v, -theta_v] * 2 +
                                                   [1 - theta_v, 1 - theta_v, theta_v, theta_v] * 2
                                                   , dim=0)  # Shape: 16 * B * 1 * H * W
            grad_flow_u = kernel * image_picked_16 * grad_bilinearcoef_flow_u  # Shape: 16, B, Ch, H, W

            # Get gradient of image pixel value wrt flow: u-direction
            grad_bilinearcoef_flow_v = torch.stack([-1 + theta_u, -1 + theta_u, 1 - theta_u, 1 - theta_u] * 2 +
                                                   [-theta_u, -theta_u, theta_u, theta_u] * 2
                                                   , dim=0)  # Shape: 16 * B * 1 * H * W
            grad_flow_v = kernel * image_picked_16 * grad_bilinearcoef_flow_v  # Shape: 16, B, Ch, H, W

            # Now, let's calculate gradients for Loss
            grad_flow = torch.sum(torch.stack([grad_flow_u, grad_flow_v], dim=0), dim=1) * grad_image_warped.unsqueeze(
                dim=0)  # Shape: 2, B, Ch, H, W
            grad_flow = torch.sum(grad_flow, dim=2)  # Shape: 2, B, H, W

            # Reshape for output
            grad_flow = torch.permute(grad_flow, [1, 0, 2, 3])

        # Requires grad does not need gradient
        # is_backward does not need gradient
        # device does not need gradient

        return grad_image, grad_kernel, grad_flow, None, None, None


from torch.autograd import gradcheck


def AdaptiveWarpingLayer_Function_gradchecker(device="cuda", set_seed=None):
    if set_seed != None:
        torch.manual_seed(0)

    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    batch = 2
    CH = 2
    H = 4
    W = 8
    input = [torch.randn(batch, CH, H, W, dtype=torch.double, requires_grad=True),
             torch.randn(batch, 16, H, W, dtype=torch.double, requires_grad=True),
             torch.randn(batch, 2, H, W, dtype=torch.double, requires_grad=True),
             True, True, device]
    test = gradcheck(AdaptiveWarpingLayer_Function.apply, input, eps=1e-6, atol=1e-4, raise_exception=True)
    print(test)