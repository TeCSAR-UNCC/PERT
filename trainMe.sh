deepspeed --master_port 12345 --include localhost:0,2 train_vae_pose.py --cfg configs/dVAE/dVAE_2d4x_res1x.yaml
