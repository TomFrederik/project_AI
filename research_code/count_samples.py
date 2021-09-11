from datasets import TrajectoryData

data_dir = '/home/lieberummaas/datadisk/minerl/data'
env_name = "MineRLTreechopVectorObf-v0"
data = TrajectoryData(env_name, data_dir)
num_samples = 0
for traj in data:
    num_samples += len(traj[0])
print(env_name, ': ', num_samples)