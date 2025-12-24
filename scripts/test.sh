
DEVICE=0
PORT=7787

# source /home/pocuser2/miniconda3/bin/activate /home/pocuser2/laptq-nedo-fed/.venv-conda

cd submodules/D-FINE


LS_MODELS=(
    /home/lap_awlv/laptq-nedo-fed/runs/data--hcmc1/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth
    /home/lap_awlv/laptq-nedo-fed/runs/data--hcmc2/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth
    # /home/lap_awlv/laptq-nedo-fed/runs/data--hcmc1--hcmc2/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth
)

LS_DATA=(
    dfine_hgnetv2_s--hcmc1
    dfine_hgnetv2_s--hcmc2
    # dfine_hgnetv2_s--hcmc1--hcmc2
)


for model in ${LS_MODELS[@]}; do
    for data in ${LS_DATA[@]}; do
        echo -e "\n\n===================================================================================="
        echo $model
        echo -e "\n\n          ************** Data: $data **************"

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