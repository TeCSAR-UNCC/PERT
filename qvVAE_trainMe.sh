deepspeed --master_port 12341 --include localhost:0,2 train_vqvae_pose.py --cfg configs/dVAE/qvVAE.yaml
