cd /home/pocuser2/laptq-nedo-fed/submodules/D-FINE

python python3 torch_inf.py \
    -c /home/pocuser2/laptq-nedo-fed/submodules/D-FINE/configs/dfine/dfine_hgnetv2_s--coco2017--lagenda.yml \
    -r /home/pocuser2/laptq-nedo-fed/runs/data--coco2017--lagenda/dfine_hgnetv2_s/train-B-E80-LR0.0002/best_stg2.pth \
    --input /home/pocuser2/datasets/OpenImageV7-v2/yolo_format/images/val \
    --device cuda:0 \
    --imgsz 640 \
    --output_dir /home/pocuser2/laptq-nedo-fed/outputs/predict