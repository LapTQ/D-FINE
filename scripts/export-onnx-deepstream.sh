python3 export_onnx_deepstream.py\
    -w outputs/train/fs26/v1.person.dfine_hgnetv2_n/best_stg1.pth \
    -c configs/dfine/dfine_hgnetv2_n--fs26.v1.person.dfine_hgnetv2_n.yml \
    --dynamic \
    --simplify \
    --size 640 \
    --opset 17 \