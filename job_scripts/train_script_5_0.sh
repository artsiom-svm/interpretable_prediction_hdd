#! /bin/bash
conda activate bash && conda activate tf2.0
python3 train.py -c config/retain_hdd_HAL_5_0.json
