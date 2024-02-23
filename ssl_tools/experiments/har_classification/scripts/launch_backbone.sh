#!/bin/bash
TRAIN_SCRIPT="train_and_test.sh"
ROOT_DATASETS_DIR="/workspaces/workdir/ssl_tools/data/view_concatenated"
RUN_ID=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="/workspaces/workdir/ssl_tools/ssl_tools/experiments/har_classification/logs/${RUN_ID}"
CWD=".."

declare -A BACKBONESCRIPTS
declare -A BACKBONEPARAMS

# BACKBONE SCRIPT
BACKBONESCRIPTS["tnc"]="my_tnc.py"
BACKBONEPARAMS["tnc"]="--mc_sample_size 20 --window_size 60 --encoding_size 10 --w 0.05 --num_classes 6"

# Run the script

# Iterate over all experiments and call the train_and_test.sh script
# for each experiment and each dataset
for experiment_name in "${!BACKBONESCRIPTS[@]}"; do
    script=${BACKBONESCRIPTS[$experiment_name]}
    args=${BACKBONEPARAMS[$experiment_name]}

    for dataset in $ROOT_DATASETS_DIR/*; do
        dataset_name=$(basename $dataset)
        ./train_backbones.sh \
            --log_dir $LOG_DIR \
            --backbone_root_data_dir $ROOT_DATASETS_DIR \
            --train_dataset $dataset_name \
            --name "${experiment_name}" \
            --script $script \
            --cwd $CWD \
            --training_mode pretrain \
            --checkpoint_metric train_loss \
            --checkpoint_metric_mode "min" \
            "${args}"
    done
done

    
    