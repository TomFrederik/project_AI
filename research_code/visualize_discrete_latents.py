import torch
import argparse
from vqvae import VQVAE
import os
import matplotlib.pyplot as plt

def main(
    vqvae_path,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    
    model = VQVAE.load_from_checkpoint(vqvae_path)
    
    num_embeddings = model.quantizer.n_embed
    print(f'{num_embeddings = }')
    
    reconstructions = []
    for i in range(num_embeddings):
        # tile latent space with embedding
        latent_idcs = torch.ones(1,16*16) * i
        latent = model.quantizer.embed_code(latent_idcs)
        print(f'{latent.shape = }')
        latent = einops.rearrange(latent, 'b (h w) c -> b c h w', h=16, w=16)
        recon = model.decode_only(latent)
        reconstructions.append(recon[0].numpy())
    
    # visualize reconstructions
    for i, recon in enumerate(reconstructions):
        plt.figure()
        plt.imshow(recon)
        plt.savefig(os.path.join(save_dir, str(i) + '.pdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae_path', type=str, default='/home/lieberummaas/datadisk/minerl/run_logs/VQVAE/MineRLTreechopVectorObf-v0/lightning_logs/version_/checkpoints/last.ckpt')
    parser.add_argument('--save_dir', type=str, default='/home/lieberummaas/datadisk/minerl/vqvae_latent_imgs')

    main(**vars(parser.parse_args()))