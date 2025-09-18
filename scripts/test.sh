
DEVICE=1
PORT=7787

# NAME_CFG=dfine_hgnetv2_s--d3.1--d3.2
# NAME_CFG=dfine_hgnetv2_s--d3-satudora
# NAME_CFG=dfine_hgnetv2_s--d5.1--d5.2
# NAME_CFG=dfine_hgnetv2_s--d7.1--d7.2

cd /home/laptq/laptq-prj-46/submodules/D-FINE
CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
    --master_port=$PORT \
    --nproc_per_node=1 \
    train.py \
    -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_s--prj57-v1.yml \
    --test-only \
    -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_s/train/best_stg2.pth \
    # -r /home/lap_awlv/laptq-nedo-fed/outputs/create-global-model-backward-update-on-epoch/dfine_hgnetv2_s/fedavg--c1-d6.1--c2-d6.2--150x1--LR0.0002/epoch149.pt \