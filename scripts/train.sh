
DEVICE=1
PORT=7780
NAME_CFG=dfine_hgnetv2_s--c1-d6.1--c2-d6.2.yml

cd /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE
CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
    --master_port=$PORT \
    --nproc_per_node=1 \
    train.py \
    -c /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE/configs/dfine/$NAME_CFG \
    --use-amp \
    --seed=0 \