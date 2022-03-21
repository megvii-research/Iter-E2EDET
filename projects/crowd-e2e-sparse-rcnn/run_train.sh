export NCCL_IB_DISABLE=1
rm -rf __pycache__
rm -rf output/events*
rm -rf output/log.txt.*
python3 train_net.py --num-gpus 8   \
    --config-file configs/50e.6h.500pro.ignore.yaml \
    --resume
