
DEVICE=0
PORT=7780
NAME_CFG=dfine_hgnetv2_s--20241122--phase-2--annotation-ver2.yml

cd /mnt/hdd10tb/Users/laptq/laptq-prj-46/submodules/D-FINE
# CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
#     --master_port=$PORT \
#     --nproc_per_node=1 \
#     train.py \
#     -c /mnt/hdd10tb/Users/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/$NAME_CFG \
#     --use-amp \
#     --seed=0 \
CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
    -c /mnt/hdd10tb/Users/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/$NAME_CFG \
    --use-amp \
    --seed=0 \