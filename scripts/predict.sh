cd /home/pocuser2/laptq-nedo-fed/submodules/D-FINE

python tools/inference/torch_inf.py \
    -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
    -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
    --input /mnt/ssd8tb/shared_workspace/prj57/datasets/thailand/videos/IMG_0904.MOV \
    --device cuda:2 \
    --imgsz 960 \
    --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input /mnt/ssd8tb/shared_workspace/prj57/datasets/thailand/videos/IMG_0905.MOV \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input /mnt/ssd8tb/shared_workspace/prj57/datasets/thailand/videos/IMG_0906.MOV \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input /mnt/ssd8tb/shared_workspace/prj57/datasets/thailand/videos/IMG_0907.MOV \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input /mnt/ssd8tb/shared_workspace/prj57/datasets/thailand/videos/IMG_0908.MOV \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms \

# =====================================
# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/01_250917/250917_Highway at Night.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/01_250917 \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input /mnt/ssd8tb/shared_workspace/prj57/datasets/uae/01_250917/250917_Mussafa_pointAtoB.MOV \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir /home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/01_250917 \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/01_正面(下村）/IMG_0667.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/01_正面(下村）" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/01_正面(下村）/IMG_0668.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/01_正面(下村）" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/01_正面(下村）/IMG_0669.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/01_正面(下村）" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/01_正面(下村）/IMG_0670.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/01_正面(下村）" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/01_正面(下村）/IMG_0672.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/01_正面(下村）" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/03_左側(檜山)/IMG_8342.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/03_左側(檜山)" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/03_左側(檜山)/IMG_8343.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/03_左側(檜山)" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/03_左側(檜山)/IMG_8344.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/03_左側(檜山)" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/03_左側(檜山)/IMG_8345.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919/03_左側(檜山)" \

# python tools/inference/torch_inf.py \
#     -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_m--prj57-v1.yml \
#     -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth \
#     --input "/mnt/ssd8tb/shared_workspace/prj57/datasets/uae/02_250919/IMG_8327.MOV" \
#     --device cuda:2 \
#     --imgsz 960 \
#     --output_dir "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms/uae/02_250919" \