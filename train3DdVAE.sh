deepspeed --master_port 12350 --include localhost:2,0 train_vae_3dpose.py --cfg configs/dVAE/dVAE_3d4x_res1x.yaml
