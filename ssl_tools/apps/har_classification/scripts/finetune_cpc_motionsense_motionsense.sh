#/bin/bash

cd ..

./train.py \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 15 \
    --batch_size 100 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/CPC_pretrain/20231201.181642/checkpoints/last.ckpt \
    --training_mode finetune \
    cpc \
    --window_size 60 \
    --num_classes 6 \
    --encoding_size 150 
