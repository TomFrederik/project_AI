import models
import torch
import numpy as np

inp = torch.rand((2,4,3,64,64))
print(f'Input tensor shape: {inp.shape}')


#############
print('\n\nTesting ConvLSTM..')
lstm_kwarg_list = [# first layer
                    {'in_channels':3, 
                    'out_channels':100, 
                    'kernel_size':3, 
                    'img_shape':(64,64)},
                    # second layer
                    {'in_channels':100, 
                    'out_channels':50, 
                    'kernel_size':3}]

CLSTM = models.ConvLSTM(lstm_kwarg_list, num_frames=4)

out = CLSTM(inp)
print(f'Output tensor shape: {out.shape}')
##############

##############
print('\n\nTesting CLSTMVAE..')
encoder_kwargs = {
                'lstm_kwarg_list':lstm_kwarg_list,
                'num_frames':4,
                'latent_dim':32
                }
decoder_kwargs = {
                'num_channels':[50,100],
                'latent_dim':32,
                'kernel_size':3
                }
VAE = models.CLSTMVAE(encoder_kwargs, decoder_kwargs)
L_reconstr, L_regul, bpd = VAE(inp)
print(f'L_reconstr = {L_reconstr},  L_regul = {L_regul}, bpd = {bpd}')
##############


##############

##############