deepspeed --master_port 12340 --include localhost:0,1,2 pretrain_peit.py --cfg configs/PERT/PEiT_panoptic_base.yaml
