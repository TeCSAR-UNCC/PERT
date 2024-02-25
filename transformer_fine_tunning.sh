deepspeed --master_port 12342 --include localhost:1,2 finetune_transformer.py --cfg configs/PERT/transformer_patch8_ntu.yaml
