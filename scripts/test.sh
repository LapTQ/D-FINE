
DEVICE=1
PORT=7787

cd submodules/D-FINE
# CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
#     --master_port=$PORT \
#     --nproc_per_node=1 \
CUDA_VISIBLE_DEVICES=$DEVICE python3 \
    train.py \
    -c /home/pocuser2/laptq-nedo-fed/submodules/D-FINE/configs/dfine/dfine_hgnetv2_s--sku.yml \
    --test-only \
    -r /home/pocuser2/laptq-nedo-fed/runs/data--sku/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth \
