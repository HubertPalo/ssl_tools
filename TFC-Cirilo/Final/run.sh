for dataset in UCI KuHar MotionSense RealWorld_thigh RealWorld_waist WISDM; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset ${dataset} --mode cls 2>&1;
done
