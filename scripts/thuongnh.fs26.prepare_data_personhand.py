import os
import json
import cv2
from tqdm import tqdm

train_f = '/mnt/hdd10tb/Users/thuongnh/datasets/satudora_mix_kita8jo_19k_v2/val.txt'
with open(train_f, 'r') as f:
    # Use rstrip('\n') to specifically target the newline
    lines = [line.rstrip('\n') for line in f]

def convert_yolo_to_coco(images_dir, labels_dir, output_json, categories):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # categories
    for i, name in enumerate(categories):
        coco["categories"].append({
            "id": i,
            "name": name
        })

    image_id = 0
    annotation_id = 0

    # image_files = [f for f in lines]

    for img_file in tqdm(lines):
        # print(img_file)
        # img_path = os.path.join(images_dir, img_file)
        img_path = os.path.join(img_file)
        l_path = os.path.join(labels_dir, img_file.rsplit('.', 1)[0] + ".txt")
        label_path = l_path.replace('images', 'labels')
        # print(label_path)

        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width = img.shape[:2]

        coco["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        # print((label_path))
        if os.path.exists(label_path):
            # print('True')
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    # print(parts)
                    if len(parts) > 5:
                        continue
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])

                    # convert to pixel
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    bbox_width = w * width
                    bbox_height = h * height

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })

                    annotation_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Saved COCO annotations to {output_json}")

categories = ["person", "hand"]  # sửa theo dataset của bạn

convert_yolo_to_coco(
    images_dir="/mnt/hdd10tb/Users/thuongnh/datasets/satudora_mix_kita8jo_19k_v2/images",
    labels_dir="/mnt/hdd10tb/Users/thuongnh/datasets/satudora_mix_kita8jo_19k_v2/labels",
    output_json="valid.json",
    categories=categories
)