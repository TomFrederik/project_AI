import numpy as np
import torch

def loss(q_values, expert_action):
    '''
    Computes the large margin classification loss J_E(Q) from the DQfD paper
    '''
    idcs = torch.arange(0,len(q_values),dtype=torch.long)
    q_values = q_values + 0.8
    q_values[idcs, expert_action] = q_values[idcs, expert_action] - 0.8
    print(q_values)
    print(f'{torch.max(q_values, dim=1)[0].shape = }')
    print(f'{q_values[idcs,expert_action].shape = }')
    return torch.max(q_values, dim=1)[0] - q_values[idcs,expert_action]

q_values = torch.rand(3,10)
expert_action = torch.as_tensor([0,1,2])
print(q_values)
print(expert_action)
print(loss(q_values, expert_action))