rm -rf __pycache__
export NCCL_IB_DISABLE=1
python3 train_net.py --num-gpus 8                               \
        --config-file configs/50e.6h.500pro.ignore.yaml         \
