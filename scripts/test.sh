
DEVICE=0
PORT=7787

# source /home/pocuser2/miniconda3/bin/activate /home/pocuser2/laptq-nedo-fed/.venv-conda

cd submodules/D-FINE


LS_MODELS=(
    # /home/pocuser2/laptq-nedo-fed/runs/data--coco2017/dfine_hgnetv2_s/train-B-E80-LR0.0002/best_stg2.pth
    # /home/pocuser2/laptq-nedo-fed/runs/data--lagenda/dfine_hgnetv2_s/train-B-E80-LR0.0002/best_stg2.pth
    # /home/pocuser2/laptq-nedo-fed/runs/data--coco2017--lagenda/dfine_hgnetv2_s/train-B-E80-LR0.0002/best_stg2.pth
    # /home/pocuser2/laptq-nedo-fed/outputs/fed/dfine_hgnetv2_s/fedavg--coco2017--lagenda--80x1--LR0.0002/epoch79.pt

    # /home/pocuser2/laptq-nedo-fed/runs/data--lagenda/dfine_hgnetv2_s/train-B-E16-LR0.0002-pOb365/best_stg2.pth
    # /home/pocuser2/laptq-nedo-fed/runs/data--virat/dfine_hgnetv2_s/train-B-E16-LR0.0002-pOb365/best_stg2.pth
    # /home/pocuser2/laptq-nedo-fed/runs/data--lagenda--virat/dfine_hgnetv2_s/train-B-E16-LR0.0002-pOb365/best_stg2.pth  
    # /home/pocuser2/laptq-nedo-fed/outputs/fed/dfine_hgnetv2_s/fedavg--lagenda--virat--16x1--LR0.0002-pOb365/epoch15.pt
    # /home/pocuser2/laptq-nedo-fed/outputs/fed/dfine_hgnetv2_s/fedavg--lagenda--virat--8x2--LR0.0002/epoch15.pt
    # /home/pocuser2/laptq-nedo-fed/dfine_s_coco.pth
    /home/pocuser2/laptq-nedo-fed/dfine_s_obj365.pth
)

LS_DATA=(
    # dfine_hgnetv2_s--coco2017--lagenda
    # dfine_hgnetv2_s--VOC2012
    # dfine_hgnetv2_s--CrowdHuman_val
    # dfine_hgnetv2_s--CityPersons
    # dfine_hgnetv2_s--d3-satudora
    # dfine_hgnetv2_s--Objects365_val
    # dfine_hgnetv2_s--virat_train
    # dfine_hgnetv2_s--openimage_val_1000
    dfine_hgnetv2_s--lagenda
    dfine_hgnetv2_s--virat
    dfine_hgnetv2_s--lagenda--virat
    dfine_hgnetv2_s--coco2017
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