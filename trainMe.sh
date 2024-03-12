deepspeed --master_port 12347 --include localhost:2,0 train_vae_pose.py --cfg configs/dVAE/dVAE_2d4x_res1x.yaml
