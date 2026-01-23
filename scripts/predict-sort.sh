cd ~/laptq-prj-46/submodules/D-FINE

python tools/inference/torch_inf-sort.py \
    -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
    -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
    --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/IMG_8327.MOV" \
    --device cuda:2 \
    --imgsz 960 \
    --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms-sort/uae/02_250919" \