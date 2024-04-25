deepspeed --master_port 12340 --include localhost:0,1,2 finetune_peit.py --cfg configs/PERT/PEiT_ucla_finetune_nano.yaml
