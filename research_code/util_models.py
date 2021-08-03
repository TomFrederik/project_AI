import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class MergeFramesWithBatch(nn.Module):
    '''
    Transforms tensor of shape (N, T, ...) to tensor of shape (N * T, ...)
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.reshape((input.shape[0] * input.shape[1], *input.shape[2:]))

class SplitFramesFromBatch(nn.Module):
    '''
    Transforms tensor of shape (N * T, ...) to tensor of shape (N, T, ...)
    '''
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
    
    def forward(self, input):
        return input.reshape((-1, self.num_frames, *input.shape[1:]))

class SplitChannelsFromClasses(nn.Module):
    '''
    Reshapes a (N, num_classes * num_channels, H, W) tensor
    into a     (N, num_classes, num_channels, H, W) tensor
    '''

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, input):
        return input.reshape((input.shape[0], -1, self.num_channels, *input.shape[2:]))

class Transpose(nn.Module):
    '''
    Transposes two dimensions of a tensor
    '''
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, input):
        return input.transpose(dim1, dim2)
