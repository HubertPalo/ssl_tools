#!/bin/bash

# Default parameters
ACCELERATOR="gpu"
DEVICES=1
PRETRAIN_BATCH_SIZE=128
TRAIN_BATCH_SIZE=128
EPOCHS=100
LEARNING_RATE=0.001
PATIENCE=30
WORKERS=8

# Print usage message
usage() {
    echo "This script trains a backbone on every dataset in the root dataset directory and saves its parameters."
    echo ""

    echo "Usage: $0 --log_dir <LOG_DIR> --backbone_root_data_dir <BACKBONE_ROOT_DATA_DIR> --train_dataset <TRAIN_DATASET> --script <SCRIPT> --name <NAME> [--cwd <CURRENT_WORKING_DIRECTORY>] <SCRIPT_ARGS>"
    echo "  --log_dir: Path to the directory where the logs will be saved"
    echo "  --backbone_root_data_dir: Path to the root directory containing the datasets for training the backbone"
    echo "  --train_dataset: Name of the dataset to use for training (e.g. KuHar)"
    echo "  --script: Path to the script to run"
    echo "  --name: Name of the experiment"
    echo "  --cwd: Path to the current working directory (default: .)"
    echo "  <SCRIPT_ARGS>: Additional arguments to pass to the script"

    echo ""

    echo "Example:"
    echo "  $0 --log_dir ../logs --backbone_root_data_dir datasets --train_dataset KuHar --script tnc_head_classifier.py --name tnc_head --cwd .. --input_size 180 --num_classes 6 --transforms fft"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --log_dir)
        LOG_DIR="$2"
        shift 2
        ;;
    --backbone_root_data_dir)
        BACKBONE_ROOT_DATA_DIR="$2"
        shift 2
        ;;
    --train_dataset)
        TRAIN_DATASET="$2"
        shift 2
        ;;
    --script)
        SCRIPT="$2"
        shift 2
        ;;
    --cwd)
        CWD="$2"
        shift 2
        ;;
    --name)
        NAME="$2"
        shift 2
        ;;
    *)
        SCRIPT_ARGS="$@"
        break
        ;;
    esac
done

# Check if required parameters are provided
if [[ -z $LOG_DIR || -z $BACKBONE_ROOT_DATA_DIR || -z $TRAIN_DATASET || -z $SCRIPT || -z $NAME ]]; then
    usage
    exit 1
fi

################################################################################

# Run the pretrain script
write_and_run_pretrain() {
    local script_file="$1"
    local output_dir="$2"
    local config_file="$output_dir/config.yaml"

    # Create output directory
    mkdir -p "$output_dir"

    # Write the config file
    echo "Writing config file...."
    echo "Pretrain data from $TRAIN_DATASET"
    python "$SCRIPT" fit \
        --print_config \
        --data "$TRAIN_DATASET" \
        --log_dir "$LOG_DIR" \
        --training_mode pretrain \
        --name "$NAME" \
        --run_id "${RUN_ID}" \
        --accelerator "$ACCELERATOR" \
        --devices "$DEVICES" \
        --batch_size "$PRETRAIN_BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --checkpoint_metric "val_loss" \
        --checkpoint_metric_mode "min" \
        --num_workers $WORKERS \
        --patience "$PATIENCE" \
        ${SCRIPT_ARGS} >"$(realpath "$config_file")"

        return_code=$?
        # Check if the script execution failed
        if [ $return_code -ne 0 ]; then
            echo "Pretrain script execution failed with return code: $return_code"
            return $return_code
        fi

        echo "Config file saved to $(realpath "$config_file")"
        # Run the script!
        echo "Running pretrain experiment from config file..."
        python "$SCRIPT" fit \
            --config "$config_file" \
            > >(tee "$output_dir/stdout.log") \
            2> >(tee "$output_dir/stderr.log" >&2)
        return_code=$?

    return $return_code
}

################################################################################

# Set default value for current working directory if not provided
CWD="${CWD:-.}"
CWD=$(realpath "$CWD")
# Resolve paths
BACKBONE_ROOT_DATA_DIR=$(realpath "$BACKBONE_ROOT_DATA_DIR")
TRAIN_DATASET="${BACKBONE_ROOT_DATA_DIR}/${TRAIN_DATASET}"
RUN_ID=$(basename "$TRAIN_DATASET")

# Output collected parameters
echo "************************* Experiment Parameters *************************"
echo "Name: $NAME - Pretrain"
echo "Log Directory: $LOG_DIR"
echo "Backbone Data Directory: $BACKBONE_ROOT_DATA_DIR"
echo "Script: $SCRIPT"
echo "Current Working Directory: $CWD"
echo "Script Args: ${SCRIPT_ARGS[@]}"
echo "Run ID: $RUN_ID"
echo "-------------------------------------------------------------------------"

# Change directory to the specified current working directory
cd "$CWD" || exit 1

# ---- PRE-TRAINING ----
# Create the pretrain output directory
PRETRAIN_OUTPUT_DIR="$LOG_DIR/pretrain/$NAME/$RUN_ID/"

# Run the pretrain script
write_and_run_pretrain "$SCRIPT" "$PRETRAIN_OUTPUT_DIR"
echo "-------------------------------------------------------------------------"

return_code=$?
if [ $return_code -ne 0 ]; then
    echo "Pretrain script execution failed with return code: $return_code"
    echo "Exiting..."
    echo "*************************************************************************"
    echo ""
    exit $return_code
fi

exit 0
