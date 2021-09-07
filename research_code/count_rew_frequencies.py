import numpy as np
import os
import argparse
from tqdm import tqdm 
import torch
import einops 

import datasets
from vecobs_vqvae import VecObsVQVAE

class Counter():
    def __init__(self, data, quantizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = data
        self.quantizer = quantizer.to(self.device)
    
        self.reward_hashmap = {2**i:i for i in range(10)}
        self.obs_counter = {2**i:[] for i in range(10)}

    def process_batch(self, batch):
        obs, _, rew, *_ = batch
        vec_obs = torch.from_numpy(obs['vector']).float().to(self.device)
        #print(f'{vec_obs.shape = }')
        z_q, ind = self.quantizer.encode_only(vec_obs)
        ind = einops.rearrange(ind, '(b l) -> b l', b=self.data.batch_size)
        
        for i, r in enumerate(rew):
            if r > 1:
                self.obs_counter[r].append(ind[i].to('cpu').numpy())

    def iter_through_data(self):
        for i, batch in tqdm(enumerate(self.data)):
            self.process_batch(batch)
            print(self.obs_counter[1024])
            #if (i+1) % 1 == 0:
                
    

def main(
    env_name, 
    data_dir, 
    log_dir, 
    batch_size,
    quantizer_version
):

    # instantiate quantizer
    quantizer_path = os.path.join(log_dir, 'VecObsVQVAE', env_name, 'lightning_logs', 'version_'+str(quantizer_version), 'checkpoints', 'last.ckpt')
    quantizer = VecObsVQVAE.load_from_checkpoint(quantizer_path)
        
    # load data
    data = datasets.BufferedBatchDataset(env_name, data_dir, batch_size, num_epochs=1)

    counter = Counter(data, quantizer)
    counter.iter_through_data()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--env_name', default='MineRLObtainIronPickaxeVectorObf-v0')
    parser.add_argument('--batch_size', default=20000, type=int)
    parser.add_argument('--quantizer_version', default=None, type=int)

    args = vars(parser.parse_args())

    main(**args)
