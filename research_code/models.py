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
        for i, kwargs in enumerate(lstm_kwarg_list):
            # if img_shape is not provided, compute it automatically:
            if 'img_shape' not in kwargs:
                
                try:
                    last_img_shape = lstm_kwarg_list[i-1]['img_shape']
                except:
                    raise ValueError('No input img shape detected. Provide img shape for input to first layer')
                
                h_new = np.floor((last_img_shape[0] - lstm_kwarg_list[i-1]['kernel_size'])/3 + 1).astype(int)
                kwargs['img_shape'] = (h_new, h_new)
                print(f'No img_shape for layer {i+1} provided. Automatically computed img_shape: {(h_new, h_new)}')

            # append modules
            model_list.append(ConvLSTMCell(**kwargs, last_only=False)) # put through CLSTM cell
            model_list.append(MergeFramesWithBatch()) # merge frames with batch for BN
            model_list.append(nn.BatchNorm2d(num_features=kwargs['out_channels'])) # BatchNorm
            model_list.append(nn.MaxPool2d(kernel_size=kwargs['kernel_size'])) # MaxPooling
            model_list.append(SplitFramesFromBatch(num_frames=num_frames)) # split frames from batch again for next iteration
        
        self.net = nn.Sequential(*model_list)
        
        h_out = np.floor((lstm_kwarg_list[-1]['img_shape'][0] - lstm_kwarg_list[-1]['kernel_size'])/3 + 1).astype(int)
        self.out_img_shape = (lstm_kwarg_list[-1]['out_channels'], h_out, h_out)

    def forward(self, input):
        return self.net(input)

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, img_shape, last_only=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_only = last_only
        self.img_shape = img_shape
        
        # define layers
        # TODO Fix padding: this pytorch version doesn't seem to support same padding --> either update or compute explicitly
        padding = (kernel_size - 1) // 2
        
        self.conv_xi = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_hi = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_ci = nn.Parameter(data=torch.zeros(img_shape))

        self.conv_xf = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_hf = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_cf = nn.Parameter(data=torch.zeros(img_shape))


        self.conv_xc = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_hc = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.conv_xo = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_ho = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_co = nn.Parameter(data=torch.zeros(img_shape))
        

        # init weight vectors
        nn.init.kaiming_normal_(self.W_ci.data, nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.W_cf.data, nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.W_co.data, nonlinearity='sigmoid')
        
            
    def forward(self, input):
        '''
        Args:
            input   - should be of shape (B, T, C, H, W)
        Returns:
            c - cell states of shape (B, T, C, H, W)
            h - cell states of shape (B, T, C, H, W)
        '''
        assert len(input.shape) == 5, f'input shape should have 5 entries, but is {input.shape}'
        
        # get sequence length T
        seq_len = input.shape[1]

        # init h and c
        h = torch.zeros((*input.shape[:2], self.out_channels, *input.shape[3:]))
        c = torch.zeros((*input.shape[:2], self.out_channels, *input.shape[3:]))

        h_0 = torch.zeros((input.shape[0], self.out_channels, *input.shape[3:]))
        c_0 = torch.zeros((input.shape[0], self.out_channels, *input.shape[3:]))

        for t in range(seq_len):
            x_t = input[:,t]
            
            if t == 0:
                i_t = torch.sigmoid(self.conv_xi(x_t) + self.conv_hi(h_0) + self.W_ci * (c_0))
                f_t = torch.sigmoid(self.conv_xf(x_t) + self.conv_hf(h_0) + self.W_cf * (c_0))
                
                c[:,t] = f_t * c_0 + i_t * torch.tanh(self.conv_xc(x_t) + self.conv_hc(h_0))
            
                
            else:
                i_t = torch.sigmoid(self.conv_xi(x_t) + self.conv_hi(h[:,t-1]) + self.W_ci * (c[:,t-1]))
                f_t = torch.sigmoid(self.conv_xf(x_t) + self.conv_hf(h[:,t-1]) + self.W_cf * (c[:,t-1]))
                
                c[:,t] = f_t * c_0 + i_t * torch.tanh(self.conv_xc(x_t) + self.conv_hc(h[:,t-1]))
        
            o_t = torch.sigmoid(self.conv_xo(x_t) + self.conv_ho(h[:,t-1]) + self.W_co) * (c[:,t])
            h[:,t] = o_t * torch.tanh(c[:,t])
        
        if self.last_only:
            return h[:,-1]
        else:
            return h

class CLSTMEncoder(nn.Module):

    def __init__(self, lstm_kwarg_list, num_frames, latent_dim):
        super().__init__()
        #TODO

        self.CLSTM = ConvLSTM(lstm_kwarg_list, num_frames)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.clstm_out_shape = self.CLSTM.out_img_shape

        self.linear = nn.Linear(np.prod(self.clstm_out_shape)*num_frames, 2*latent_dim)
    
    def forward(self, x):
        '''
        x - input, for mineRL (B, T, C, H, W) batch of stack of frames
        '''
        clstm_out = self.CLSTM(x)
        latent_out = self.linear(self.flatten(clstm_out))

        mean, log_std = torch.chunk(latent_out, chunks=2, dim=-1)
        
        return mean, log_std

class ConvDecoder(nn.Module):

    def __init__(self, latent_dim, clstm_out_shape, num_channels, kernel_size):
        '''
        Args:
            latent_dim - dimension of the latent space
            clstm_out_shape - shape of an image after it is passed through the encoding CLSTM, should be of form (C, H, W)
            num_channels - list of ints describing the number of channels after each convolution. 
                           len(num_channels) is used to determine number of layers.
        '''
        super().__init__()
        # MAYBE?
        # output needs to passed through 256-way softmax to encode for pixels
        
        self.clstm_out_shape = clstm_out_shape

        # compute padding for same-padding
        padding = (kernel_size - 1) // 2

        # compute the upsample factor used in each layer
        up_factor = np.exp(np.log(64 / clstm_out_shape[1]) / len(num_channels))
        print(f'Upsample factor in ConvDecoder: {up_factor:.2f}')
        img_sizes = [clstm_out_shape[1] * up_factor**i for i in range(1,len(num_channels))]
        img_sizes.append(64)
        img_sizes = np.floor(img_sizes).astype(int)
        print(f'calculated image sizes during deconv: {img_sizes}')

        self.linear = nn.Sequential(nn.Linear(latent_dim, np.prod(clstm_out_shape)), nn.GELU())
        
        num_channels = [clstm_out_shape[0]] + num_channels

        layer_list = []
        for i in range(len(num_channels)-1):
            layer_list.append(nn.Upsample((img_sizes[i], img_sizes[i])))
            layer_list.append(nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=kernel_size, padding=padding))

        layer_list.append(nn.Conv2d(num_channels[-1], 3, kernel_size=kernel_size, padding=padding))
        
        self.deconv = nn.Sequential(*layer_list)
    
    def forward(self, z):
        '''
        z - latent vector
        '''
        out = self.linear(z)
        out = out.reshape((z.shape[0], *self.clstm_out_shape))
        out = self.deconv(out)



class CLSTMVAE(nn.Module):

    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__()
        self.encoder = CLSTMEncoder(**encoder_kwargs)
        self.decoder = ConvDecoder(**decoder_kwargs, clstm_out_shape=self.encoder.clstm_out_shape)
        #TODO
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

    def sample(self, mean, log_std):
        '''
        Implements reparameterizatio trick to sample from the given normal distribution
        '''
        z = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(mean), torch.ones_like(log_std))
        return z
    
    def forward(self, x):
        '''
        x - input, for mineRL (B, T, C, H, W) batch of stack of frames
        '''
        # encode
        mean, log_std = self.encoder(x)

        # compute KL difstance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1)

        # sample latent vector
        z = self.sample(mean, log_std)

        # decode
        x_new = self.decoder(z)
        
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


