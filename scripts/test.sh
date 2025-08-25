
DEVICE=3
PORT=7787

# NAME_CFG=dfine_hgnetv2_s--d3.1--d3.2
# NAME_CFG=dfine_hgnetv2_s--d3-satudora
# NAME_CFG=dfine_hgnetv2_s--d5.1--d5.2
NAME_CFG=dfine_hgnetv2_s--d7.1--d7.2

cd /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE
CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
    --master_port=$PORT \
    --nproc_per_node=1 \
    train.py \
    -c /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE/configs/dfine/$NAME_CFG.yml \
    --test-only \
    -r /home/lap_awlv/laptq-nedo-fed/runs/data--c1-d6.1--c2-d6.2/dfine_hgnetv2_s/train-B-E150-LR0.0002/last.pth \
    # -r /home/lap_awlv/laptq-nedo-fed/outputs/create-global-model-backward-update-on-epoch/dfine_hgnetv2_s/fedavg--c1-d6.1--c2-d6.2--150x1--LR0.0002/epoch149.pt \