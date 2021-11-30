import torch
from modules.AdaptiveWarpingLayer import AdaptiveWarpingLayer, AdaptiveWarpingLayer_Function_gradchecker

########### GRADCHECK ############
print("Gradchecking...")
AdaptiveWarpingLayer_Function_gradchecker()



########### CHECK OUTPUT ############
device = "cuda"
# torch.manual_seed(0)
if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
batch = 4
CH = 99
H = 4
W = 8

# Random Input tensors (for gradcheck)
# image = torch.randn(batch,3,256,448,dtype=torch.float32, requires_grad=True)
depth = torch.randn(batch,1,256,448,dtype=torch.float32, requires_grad=True) + 2.0
context = torch.randn(batch, CH, 256, 448, dtype=torch.float32, requires_grad=True)
kernel = torch.randn(batch,16,256,448,dtype=torch.float32,requires_grad=True)
flow = torch.randn(batch,2,256,448,dtype=torch.float32,requires_grad=True)


print("Declaring model...")
AWL_module = AdaptiveWarpingLayer(device=device).to(device)

# Context image
print("Checking output for channel 195 context image...")
input = [context,
        #  depth,
        #  context,
         kernel,
         flow,
         True, True, device]

warped_context = AWL_module(context, kernel, flow)
print("warped_context shape and dtype: ", warped_context.shape, warped_context.dtype)

# Depth image
print("Checking output for channel 195 context image...")
input = [depth,
        #  depth,
        #  context,
         kernel,
         flow,
         True, True, device]

warped_depth = AWL_module(depth, kernel, flow)
print("warped_depth shape and dtype: ", warped_depth.shape, warped_depth.dtype)
