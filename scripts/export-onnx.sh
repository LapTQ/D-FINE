cd ~/laptq-prj-46/submodules/D-FINE

python tools/deployment/export_onnx.py \
    -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
    -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
    --check \
    --simplify \
    --imgsz 960 \
    --batch_size 1 \