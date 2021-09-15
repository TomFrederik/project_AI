import torch.nn as nn
import torch
import einops



B = 2
T = 8
input_tensor = torch.ones((B, 32, T-1, 16, 16), dtype=torch.float)

cnn_3d_1 = nn.Sequential(
    nn.Conv3d(32, 64, 3, 1, 1),
    nn.ReLU(),
    nn.AdaptiveMaxPool3d((16,16,T-1)),
    nn.Conv3d(64, 64, 3, 1, 1),
    nn.ReLU(),
    nn.AdaptiveMaxPool3d((8,8, T-1)),
    nn.Conv3d(64, 64, 3, 1, 1),
    nn.ReLU(),
    nn.AdaptiveMaxPool3d((4,4, T-1)), # B L 256 4 4
)    

linear = nn.Linear((T-1)*4*4*64, 64*8*8)

cnn_3d_2 = nn.Sequential(
    nn.ConvTranspose2d(64, 64, 3, 1, 1),
    nn.UpsamplingNearest2d((16,16)),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 64, 3, 1, 1),
    nn.UpsamplingNearest2d((32,32)),
    nn.ConvTranspose2d(64, 64, 3, 1, 1),
    nn.UpsamplingNearest2d((16,16)),
    nn.ConvTranspose2d(64, 32, 3, 1, 1)
)

print(f'{input_tensor.shape}')
out1 = cnn_3d_1(input_tensor)
print(f'{out1.shape}')
out1 = einops.rearrange(out1, 'b t c h w -> b (t c h w)')
print(f'{out1.shape}')
out2 = linear(out1)
print(f'{out2.shape}')
out2 = einops.rearrange(out2, 'b (c h w) -> b c h w', c=64, h=8, w=8)
print(f'{out2.shape}')
out3 = cnn_3d_2(out2)
print(f'{out3.shape}')
