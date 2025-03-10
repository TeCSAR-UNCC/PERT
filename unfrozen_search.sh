#!/bin/bash

# Define the range of values for abnormal_w, weight_decay, and aug_abnormal
abnormal_ws=(1 2 3)
weight_decays=(0.0003 0.0006)
aug_abnormals=(10 15)

# Initialize the counter for custom_run_name
counter=1

# Set the complete path to the YAML config file
config_file="/home/galinezh/PERT/configs/PERT/PEiT_UBnormal_Unforzen_finetune_base_3d.yaml"

# Iterate over all combinations of abnormal_w, weight_decay, and aug_abnormal
for abnormal_w in "${abnormal_ws[@]}"; do
    for weight_decay in "${weight_decays[@]}"; do
        for aug_abnormal in "${aug_abnormals[@]}"; do
            # Update the YAML configuration file with the new parameters

            # Create a backup of the original YAML file
            cp $config_file "${config_file}.bak"

            # Use sed to update the abnormal_w, weight_decay, aug_abnormal, and custom_run_name in the YAML file
            sed -i "s/abnormal_w: .*/abnormal_w: $abnormal_w/" $config_file
            sed -i "s/weight_decay: .*/weight_decay: $weight_decay/" $config_file
            sed -i "s/aug_abnormal: .*/aug_abnormal: $aug_abnormal/" $config_file
            sed -i "s/custom_run_name: .*/custom_run_name: \"UBnormal_unfrozen_search_$counter\"/" $config_file

            # Increment the counter
            counter=$((counter + 1))

            # Run the training with the modified YAML file
            python3 finetune_anomaly.py --cfg $config_file

            # Optionally, move the output (e.g., logs, models) to a specific folder named after the run
            # mv output/ "output_UBnormal_search_$((counter - 1))"

            # Restore the original YAML file for the next iteration
            mv "${config_file}.bak" $config_file
        done
    done
done
