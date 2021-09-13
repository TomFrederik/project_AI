import torch.nn as nn
import torch

T = 3
B = 2
D_in = 2
D_out = 10

gru = nn.GRU(input_size=D_in, hidden_size=D_out, batch_first=True).to('cuda')

input_tensor = torch.ones((B,T,D_in), requires_grad=True, device='cuda')
h_0 = torch.ones((1,B,D_out), requires_grad=True, device='cuda')

output1, _ = gru(input_tensor)
loss1 = output1.pow(2).sum()
loss1.backward()
print(gru.weight_hh_l0.grad)

output2, _ = gru(input_tensor, h_0)
loss2 = output2.pow(2).sum()
loss2.backward()
print(gru.weight_hh_l0.grad)