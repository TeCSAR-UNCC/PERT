deepspeed --master_port 12340 --include localhost:2,1 finetune_peit.py --cfg configs/PERT/PEiT_ucla_finetune_base.yaml
