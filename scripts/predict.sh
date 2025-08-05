cd /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE

python tools/inference/torch_inf.py \
    -c /home/lap_awlv/laptq-nedo-fed/submodules/D-FINE/configs/dfine/dfine_hgnetv2_s--c1.yml \
    -r /home/lap_awlv/laptq-nedo-fed/runs/data--c1/dfine_hgnetv2_s/train-B-E150-LR0.0002/last.pth \
    --input /home/lap_awlv/laptq-nedo-fed/data/detection_people_pseudo/batch1/B8-A4-4F-D2-F8-3A/images/2025_03_18/1742296344124_30000.jpg \
    --device cuda:0