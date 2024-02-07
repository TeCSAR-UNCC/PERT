deepspeed --master_port 12344 --include localhost:0,1 train_vae_pose.py --cfg configs/dVAE/dVAE_2d4x_res1x.yaml
