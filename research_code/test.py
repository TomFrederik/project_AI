from datasets import TrajectoryData
from torch.utils.data import DataLoader

env_name = 'MineRLTreechopVectorObf-v0'
data_dir = '/home/lieberummaas/datadisk/minerl/data'
data = TrajectoryData(env_name, data_dir)
data_loader = DataLoader(data, batch_size=None, num_workers=0)


for batch_idx, batch in enumerate(data_loader):
    print(batch_idx)
