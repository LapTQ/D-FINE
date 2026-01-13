cd /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE

python tools/inference/torch_inf.py \
    -c /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE/configs/dfine/dfine_hgnetv2_s--hcmc2.yml \
    -r /home/lap_awlv/laptq-nedo-fed/runs/data--hcmc2/dfine_hgnetv2_s/train-B-E60-LR0.0002/best_stg2.pth \
    --input /home/lap_awlv/laptq-nedo-fed/data/satudora_det_normal/test/images \
    --device cuda:1 \
    --imgsz 640 \
    --output_dir /home/lap_awlv/laptq-nedo-fed/outputs/predict2