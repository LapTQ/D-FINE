
DEVICE=1
PORT=7787

source /home/pocuser2/miniconda3/bin/activate /home/pocuser2/laptq-nedo-fed/.venv-conda

cd submodules/D-FINE


LS_MODELS=(
    /home/pocuser2/laptq-nedo-fed/runs/data--sku/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth
    /home/pocuser2/laptq-nedo-fed/runs/data--locount/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth
    /home/pocuser2/laptq-nedo-fed/runs/data--sku--locount/dfine_hgnetv2_s/train-B-E80-LR0.0002/best_stg2.pth
)

LS_DATA=(
    dfine_hgnetv2_s--sku
    dfine_hgnetv2_s--locount
    dfine_hgnetv2_s--sku--locount
)


for model in ${LS_MODELS[@]}; do
    for data in ${LS_DATA[@]}; do
        # CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
        #     --master_port=$PORT \
        #     --nproc_per_node=1 \
        CUDA_VISIBLE_DEVICES=$DEVICE python3 \
            train.py \
            -c configs/dfine/$data.yml \
            --test-only \
            -r $model \
    done
done