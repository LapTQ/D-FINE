python tools/inference/torch_inf-sort.py \
    -c configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
    -r runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
    --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/IMG_8327.MOV" \
    --device cuda:2 \
    --imgsz 960 \
    --output_dir "outputs/predict_results/dfinem-960-nms-sort/uae/02_250919" \