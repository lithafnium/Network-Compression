#!/bin/bash
eval "$(conda shell.bash hook)"
echo "Activating cs222"
conda activate cs222
echo "Redirecting into proper directory"
cd /data/leonardtang/cs222proj
echo "Starting training and evaluation..."
python ./main.py --data-type=small-world