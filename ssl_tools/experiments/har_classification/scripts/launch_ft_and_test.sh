#!/bin/bash
TRAIN_SCRIPT="train_and_test.sh"
ROOT_DATASETS_DIR="/workspaces/workdir/ssl_tools/data/standartized_balanced"
ROOT_BACKBONES_DIR="/workspaces/workdir/ssl_tools/ssl_tools/experiments/har_classification/logs/dim10/pretrain/tnc"
RUN_ID="dim10"
LOG_DIR="/workspaces/workdir/ssl_tools/ssl_tools/experiments/har_classification/logs/${RUN_ID}"
CWD=".."

declare -A MODELSCRIPTS
declare -A MODELPARAMS

# Linear with finetuning
MODELSCRIPTS["lin_ft"]="linear.py"
MODELPARAMS["lin_ft"]="--encoding_size 10 --num_classes 6 --update_backbone"

# Linear with frozen backbone
MODELSCRIPTS["lin"]="linear.py"
MODELPARAMS["lin"]="--encoding_size 10 --num_classes 6"

# MLP with finetuning
MODELSCRIPTS["mlp_ft"]="tnc_default.py"
MODELPARAMS["mlp_ft"]="--encoding_size 10 --num_classes 6 --update_backbone"

# MLP with frozen backbone
MODELSCRIPTS["mlp"]="tnc_default.py"
MODELPARAMS["mlp"]="--encoding_size 10 --num_classes 6"

# Run the script

# Iterate over all experiments and call the train_and_test.sh script
# for each experiment and each dataset
for backbone in $ROOT_BACKBONES_DIR/*; do
    backbone_name=$(basename $backbone)

    for experiment_name in "${!MODELSCRIPTS[@]}"; do
        script=${MODELSCRIPTS[$experiment_name]}
        args=${MODELPARAMS[$experiment_name]}

        for dataset in $ROOT_DATASETS_DIR/*; do
            dataset_name=$(basename $dataset)
            ./finetune_and_test.sh \
                --log_dir $LOG_DIR \
                --root_dataset_dir $ROOT_DATASETS_DIR \
                --train_dataset $dataset_name \
                --load_backbone $ROOT_BACKBONES_DIR/${dataset_name}/checkpoints/last.ckpt \
                --name "${experiment_name}" \
                --script $script \
                --cwd $CWD \
                "${args}"
        done
    done
done