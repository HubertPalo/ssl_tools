from pathlib import Path
import sys
sys.path.append('../')


class Config():
    # Run parameters
    num_epochs = 1000
    accelerator = "gpu"
    num_gpus = 1
    strategy = "ddp"
    checkpoint_dir = Path("save_models")
    tfc_pretrain_checkpoint = checkpoint_dir
    results_tfc_classifier = Path("results")
    batch_size = 128

    # NXTent Loss
    jitter_ratio = 2
    length_alignment = 360
    drop_last = True
    num_workers = 10
    learning_rate = 3e-4
    temperature = 0.2
    use_cosine_similarity = True
    is_subset = True
    
    def __init__(self):
        pass