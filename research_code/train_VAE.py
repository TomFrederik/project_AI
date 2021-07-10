import models
import argparse
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

def train_VAE(args):
    
    kernel_size = 3
    num_frames = 4
    latent_dim = 32
    num_channels = [50,100] # channels for decoder

    lstm_kwarg_list = [
        # first layer
        {
            'in_channels':3,
            'out_channels':100,
            'kernel_size':kernel_size,
            'img_shape':(64,64)
        },
        # second layer
        {
            'in_channels':100,
            'out_channels':50,
            'kernel_size':kernel_size
        }
    ]

    encoder_kwargs = {
        'lstm_kwarg_list':lstm_kwarg_list,
        'num_frames':num_frames,
        'latent_dim':latent_dim
    }
    decoder_kwargs = {
        'latent_dim':latent_dim,
        'num_channels':num_channels,
        'kernel_size':kernel_size
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.CLSTMVAE(encoder_kwargs, decoder_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    writer = SummaryWriter() #TODO: add logdir arg

    # create data
    data = datasets.OfflineData(env, num_samples, num_frames, kmeans)
    lengths = [int(len(data)*(1-args.val_perc)), int(len(data)*args.val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)
    
    for epoch in range(args.epochs):
        print(f'\n\nEpoch {epoch+1}:')

        for i, (_, img, _, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            _, _, bpd = model(img.to(device)).mean()
            bpd.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        for i, (_, img, _, _, _, _) in enumerate(val_loader):
            with torch.no_grad():
                _, _, bpd = model(img.to(device)).mean()
        writer.add_scalar('Validation/bpd',bpd.item())

        # every N epochs, display result of reconstruction
        if (epoch+1) % args.eval_freq:
            # sample a random val image
            id = np.random.choice(len(val_data))
            _, img, _, _, _, _ = val_data[id]
            with torch.no_grad():
                rec_img = model.reconstruct_only(img.to(device))
            writer.add_images('Reconstruction', torch.stack([img, rec_img],dim=0),global_step=epoch+1)
    
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--lr', default=3e-4, help='Learning rate')
    parser.add_argument('--val_perc', default=0.1, help='How much of the data should be used for validation')
    parser.add_argument('--eval_freq', default=1, help='How often to reconstruct a random val image for tensorboard')

    args = parser.parse_args()

    train_VAE(args)
