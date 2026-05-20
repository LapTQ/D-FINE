python3 tools/inference/torch_inf.py \
    -c configs/dfine/dfine_hgnetv2_s--fs26-personhand.yml \
    -r outputs/train/fs26/v1.dfine_hgnetv2_s.person_hand/best_stg1.pth \
    --input /home/laptq/laptq-fs26-shoplifting-detection/data/test_sat_personhand/images \
    --device cuda:5 \
    --imgsz 640 \
    --output_dir outputs/predict