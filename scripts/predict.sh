cd ~/laptq-prj-46/submodules/D-FINE

python tools/inference/torch_inf.py \
    -c /home/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/dfine_hgnetv2_s--prj57-v1.yml \
    -r /home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_s/train/best_stg2.pth \
    --input /mnt/ssd8tb/shared_workspace/prj57/datasets/thailand/videos/IMG_0908.MOV \
    --device cuda:1 \
    --imgsz 960 \