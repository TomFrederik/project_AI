import torch
import torch.nn as nn
import numpy as np

class MergeFramesWithBatch(nn.Module):
    '''
    Transforms tensor of shape (N, T, ...) to tensor of shape (N * T, ...)
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.view((input.shape[0] * input.shape[1], *input_shape[2:]))

class SplitFramesFromBatch(nn.Module):
    '''
    Transforms tensor of shape (N * T, ...) to tensor of shape (N, T, ...)
    '''
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
    
    def forward(self, input):
        return input.view((-1, self.num_frames, *input_shape[2:]))

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

class ConvLSTM(nn.Module):

    def __init__(self, lstm_kwarg_list, num_frames):
        super().__init__()
        model_list = []
        for kwargs in lstm_kwarg_list:
            model_list.append(ConvLSTMCell(**kwargs)) # put through CLSTM cell
            model_list.append(MergeFramesWithBatch()) # merge frames with batch for BN
            model_list.append(nn.BatchNorm2d(num_features=kwargs['out_channels'])) # BatchNorm
            model_list.append(nn.MaxPooling) # MaxPooling
            model_list.append(SplitFramesFromBatch(num_frames=num_frames)) # split frames from batch again for next iteration
        
        self.net = nn.Sequential(**model_list)

    def forward(self, input):
        return self.net(input)

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        #TODO: fill in shapes
        self.conv_xi = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_hi = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.W_ci = torch.zeros_like()

        self.conv_xf = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_hf = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.W_cf = torch.zeros_like()


        self.conv_xc = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_hc = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        
        self.conv_xo = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_ho = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.W_co = torch.zeros_like()
        

        # init weight vectors
        # TODO
            
    def forward(self, input):
        '''
        Args:
            input   - should be of shape (B, T, C, H, W)
        Returns:
            c - cell states of shape (B, T, C, H, W)
            h - cell states of shape (B, T, C, H, W)
        '''
        assert len(input.shape) == 5, f'input shape should have 5 entries, but is {input.shape}'
        
        seq_len = input.shape[0]
        h = torch.zeros_like(input)
        c = torch.zeros_like(input)

        h_0 = torch.zeros((input.shape[0], *input.shape[2:]))
        c_0 = torch.zeros((input.shape[0], *input.shape[2:]))

        for t in range(seq_len):
            x_t = input[:,t]
            
            if t == 0:
                i_t = torch.sigmoid(self.conv_xi(x_t) + self.conv_hi(h_0) + torch.diag_embed(self.W_ci)(c_0))
                f_t = torch.sigmoid(self.conv_xf(x_t) + self.conv_hf(h_0) + torch.diag_embed(self.W_cf)(c_0))
                
                c[:,t] = f_t * c_0 + i_t * torch.tanh(self.conv_xc(x_t) + self.conv_hc(h_0))
            
                
            else:
                i_t = torch.sigmoid(self.conv_xi(x_t) + self.conv_hi(h[:,t-1]) + torch.diag_embed(self.W_ci)(c[:,t-1]))
                f_t = torch.sigmoid(self.conv_xf(x_t) + self.conv_hf(h[:,t-1]) + torch.diag_embed(self.W_cf)(c[:,t-1]))
                
                c[:,t] = f_t * c_0 + i_t * torch.tanh(self.conv_xc(x_t) + self.conv_hc(h[:,t-1]))
        
            o_t = torch.sigmoid(self.conv_xo(x_t) + self.conv_ho(h[:,t-1]) + torch.diag_embed(self.W_co)(c[:,t]))
            h[:,t] = o_t * torch.tanh(c[:,t])
        
        # return c and h
        return c, h

class CLSTM_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        #TODO
    
    def forward(self, x, y):
        '''
        x - input, for mineRL (B, T, C, H, W) batch of stack of frames
        y - conditioning input, for mineRL this is the (B, 64) obs tensor
        '''
        out = self.net(x)
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)
        
        return mean, log_std

class CLSTM_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        #TODO
        # output needs to passed through 256-way softmax to encode for pixels
    
    def forward(self, z, y):
        '''
        z - latent vector
        y - conditioning vector, for mineRL this is the (B, 64) obs tensor
        '''
        #TODO


class CLSTM_cVAE(nn.Module):

    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__()
        self.encoder = CLSTM_Encoder(**encoder_kwargs)
        self.decoder = CLSTM_Decoder(**decoder_kwargs)
        #TODO
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

    def sample(self, mean, log_std):
        '''
        Implements reparameterizatio trick to sample from the given normal distribution
        '''
        z = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(mean), torch.ones_like(log_std))
        return z
    
    def forward(self, x, y):
        '''
        x - input, for mineRL (B, T, C, H, W) batch of stack of frames
        y - conditioning vector, for mineRL this is the (B, 64) obs tensor
        '''
        # encode
        mean, log_std = self.encoder(x, y)

        # compute KL difstance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1)

        # sample latent vector
        z = self.sample(mean, log_std)

        # decode
        x_new = self.decoder(z, y)
        
        # convert x to classes
        x = x.uint8()

        # compute reconstruction loss, sum over all dimension except batch
        L_reconstr = self.CE_loss(x, x_new).sum(dim=list(range(1,len(x.shape))))

        # compute elbo
        elbo = - L_reconstr + L_regul

        # convert into bits per dimension loss
        bpd = elbo * np.log2(np.exp(1)) / np.prod(x.shape[1:])

        # take means
        L_reconstr = torch.mean(L_reconstr)
        L_regul = torch.mean(L_regul)
        
        return L_reconstr, L_regul, bpd


