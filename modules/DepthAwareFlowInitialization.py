import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DepthAwareFlowInitialization(nn.Module):
    def __init__(self, device, backward=True):
        super(DepthAwareFlowInitialization, self).__init__()
        self.device = device
        self.backward = backward # If backward is False: use I(t-1) -> I(t) to initialize I(t) -> I(t+1)
                                 # If backward is True: use I(t) -> I(t-1) to initialize I(t+1) -> I(t)
    def forward(self, flow, inv_depth): # flow must be B, 2, H, W / inv_depth must be B, 1, H, W
        return DepthAwareFlowInitialization_Function.apply(flow,inv_depth, flow.requires_grad, self.backward, self.device)

class DepthAwareFlowInitialization_Function(Function):
    def __init__(self):
        super(DepthAwareFlowInitialization_Function,self).__init__()

    @staticmethod
    def forward(ctx, flow, inv_depth, requires_grad, is_backward, device): 
        ## Recall: If is_backward is False, flow is flow from I(t-1) -> I(t), inv_depth is of I(t)
        ## Recall: weight is inverse of depth

        # flow shape: B, 2, H, W    /    inv_depth shape: B, 1, H, W (Already unsqueezed)
        B, _, H, W = inv_depth.shape
        
        # Define meshgrid
        y_grid, x_grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        mesh_before = torch.stack([x_grid, y_grid], dim=0)

        # After flow, where are the pixels? [Getting S(x)]
        if not is_backward:
            if device == "cuda":
                mesh_after = torch.round(mesh_before + flow).type(torch.cuda.LongTensor) # Shape: B, 2, H, W
            elif device == "cpu":
                mesh_after = torch.round(mesh_before + flow).type(torch.LongTensor)
        else:
            if device == "cuda":
                mesh_after = torch.round(mesh_before - flow).type(torch.cuda.LongTensor) # Shape: B, 2, H, W
            elif device == "cpu":
                mesh_after = torch.round(mesh_before - flow).type(torch.LongTensor)

        # Boolean array checking if in_range (shape: 4, B, H, W)
        in_range = torch.stack([mesh_after[:,0] >= 0, mesh_after[:,0] < W, mesh_after[:,1] >= 0, mesh_after[:,1] < H], dim=0)
        in_range = torch.all(in_range, dim=0)

        # Only consider "future is in-range" pixels
        mesh_after = in_range.unsqueeze(dim=1) * mesh_after
        # torch.where(in_range.unsqueeze(dim=1), mesh_after, 0)


        # What is weights?
        weights = in_range.unsqueeze(dim=1) * inv_depth 
        # weights = torch.where(in_range, inv_depth, 0)


        # What is weighted flow of next step? [Getting sum(weight * flow) of before]
        weighted_flow = flow * weights


        # Ravel
        unravel_length = B*H*W
        if B == 1:
            mesh_after_raveled = torch.reshape(mesh_after[:,0] + mesh_after[:,1] * W, [unravel_length]) # Change to 1D
        else:
            mesh_after_raveled = torch.reshape(mesh_after[:,0] + mesh_after[:,1] * W + torch.arange(start=0, end=B).unsqueeze(dim=-1).unsqueeze(dim=-1) * H * W, [unravel_length])  # Change to 1D
        
        
        weighted_flow_raveled = torch.reshape(weighted_flow.permute(1,0,2,3), [2, unravel_length])
        weights_raveled = torch.reshape(weights, [unravel_length])


        # Now, accumulate to current I(t)
        flow_x = torch.bincount(input=mesh_after_raveled, weights=weighted_flow_raveled[0],minlength=unravel_length)
        flow_y = torch.bincount(input=mesh_after_raveled, weights=weighted_flow_raveled[1],minlength=unravel_length)
        w_f = torch.bincount(input=mesh_after_raveled, weights=weights_raveled, minlength=unravel_length)
        

        #
        w_f = w_f[0:unravel_length].unsqueeze(dim=0)
        # w_f = torch.clamp(w_f, min=1e-7).type(w_f.dtype)
        w_f = w_f + 1e-7
        inv_sum_weight = 1.0 / w_f
        

        #
        flow_est = torch.stack([flow_x[0:unravel_length], flow_y[0:unravel_length]], dim=0)
        inv_sum_weight = (flow_est[0] != 0) * inv_sum_weight
        holes_unfilled = flow_est * inv_sum_weight
        depth_aware_flow_initialization = torch.permute(torch.reshape(holes_unfilled, [2, B, H, W]), dims=[1,0,2,3])

        # Save for backward
        ctx.save_for_backward(flow, holes_unfilled, weights_raveled , inv_sum_weight, mesh_after_raveled)
    
        return depth_aware_flow_initialization

    @staticmethod
    def backward(ctx, grad_depth_aware_flow):
        
        # Retrieve saved tensors from forward
        flow, holes_unfilled, weights_raveled, inv_sum_weight, mesh_after_raveled = ctx.saved_tensors # RECALL: weights are inv_depth of ones who does not get out of range in future
        grad_flow = grad_inv_depth = None

        # Get shape
        B, _, H, W= flow.shape
        # print(flow.shape)


        # Ravel gradient of depth aware flow (we inputted)
        # print(grad_depth_aware_flow.shape)
        grad_depth_aware_flow = torch.reshape(torch.permute(grad_depth_aware_flow, [1,0,2,3]), [2,B*H*W])
        # print("flow, holes_unfilled, weights_raveled, inv_sum_weight, mesh_after_raveled shapes", flow.shape, holes_unfilled.shape, weights_raveled.shape, inv_sum_weight.shape, mesh_after_raveled.shape)


        if ctx.needs_input_grad[0]:
            # print(mesh_after_raveled.dtype)
            # print(grad_depth_aware_flow.shape)
            # print(grad_depth_aware_flow[0,mesh_after_raveled])
            # print(inv_sum_weight.shape)
            # print(inv_sum_weight[:, mesh_after_raveled])
            # gdaf = grad_depth_aware_flow[:,mesh_after_raveled]
            # isw = inv_sum_weight[:, mesh_after_raveled]
            # wru = weights_raveled.unsqueeze(dim=0)
            # grad_flow = gdaf * isw * wru
            grad_flow = grad_depth_aware_flow[:, mesh_after_raveled] * inv_sum_weight[:, mesh_after_raveled] * weights_raveled.unsqueeze(dim=0)
            grad_flow = torch.permute(torch.reshape(grad_flow, [2, B, H, W]), [1,0,2,3])
            # print(grad_flow.shape)


        if ctx.needs_input_grad[1]:
            flow_raveled = torch.reshape(flow.permute(1,0,2,3), [2,B*H*W])
            # print(flow_raveled.shape)
            # hu = holes_unfilled[:,mesh_after_raveled]
            # print(hu.shape)
            # flow_diff = flow_raveled - hu
            # print(flow_diff.shape)
            # x = flow_diff * inv_sum_weight[:, mesh_after_raveled]
            # print(x.shape)
            # grad_inv_depth = grad_depth_aware_flow[:, mesh_after_raveled] * x
            # grad_inv_depth = grad_inv_depth * (weights_raveled.unsqueeze(dim=0) != 0)
            grad_inv_depth = (flow_raveled - holes_unfilled[:, mesh_after_raveled]) * inv_sum_weight[:, mesh_after_raveled] * grad_depth_aware_flow[:, mesh_after_raveled] * (weights_raveled.unsqueeze(dim=0) != 0)
            grad_inv_depth = torch.sum(grad_inv_depth, dim=0)
            # grad_inv_depth = (flow_raveled - holes_unfilled[mesh_after_raveled]) * inv_sum_weight[mesh_after_raveled].unsqueeze(dim=0) * (weights_raveled.unsqueeze(dim=0) != 0)
            grad_inv_depth = torch.reshape(grad_inv_depth, [B, H, W])
            grad_inv_depth = grad_inv_depth.unsqueeze(dim=1)
            # print(grad_inv_depth.shape)


        # Requires grad does not need gradient
        # is_backward does not need gradient
        # device does not need gradient

        return grad_flow, grad_inv_depth, None, None, None
        
    
        
    


from torch.autograd import gradcheck


def DepthAwareFlowInitialization_Function_gradchecker(device="cuda", set_seed=None):
    if set_seed != None:
        torch.manual_seed(0)

    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    batch = 4
    input = [torch.randn(batch, 2, 12, 26, dtype=torch.double, requires_grad=True),
             torch.randn(batch, 1, 12, 26, dtype=torch.double, requires_grad=True), True, True, device]
    test = gradcheck(DepthAwareFlowInitialization_Function.apply, input, eps=1e-6, atol=1e-4, raise_exception=True)
    print(test)
