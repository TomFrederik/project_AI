

# train VAE
python3 -u train_VAE.py --data_dir $MRL/data/numpy_data --env_name MineRLObtainIronPickaxeVectorObf-v0 --batch_size 150 --epochs 1 --log_dir $MRL/run_logs --model_class Conv

# train Dynamics
python3 -u train_Dynamics --data_dir $MRL/data/numpy_data --env_name MineRLObtainIronPickaxeDenseVectorObf-v0 --batch_size 100 --epochs 1 --log_dir $MRL/run_logs --model_class rssm --model_path
