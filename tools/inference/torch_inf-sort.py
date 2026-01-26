"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


import cv2

COLORS = np.random.randint(0, 256, size=(32, 3), dtype=np.int32)


def render_tracks(img, track_results, id_class, map_id_to_name):
    for track in track_results:
        track_id = int(track.track_id)
        x1_p, y1_p, x2_p, y2_p, score, _ = track.box
        color = tuple(COLORS[track_id % len(COLORS)].tolist())
        cv2.rectangle(
            img, (int(x1_p), int(y1_p)), (int(x2_p), int(y2_p)), color, thickness=2
        )
        cv2.putText(
            img=img,
            text=f"{map_id_to_name[id_class]} {track_id} {score:.2f}",
            org=(int(x1_p + 3), int(y1_p - 5)),
            thickness=2,
            color=color,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
        )


from sort import Sort
from detection_info import DetectionInfo


# LapTQ: TODO depends on videos
def preprocess_frame(frame):
    frame = np.ascontiguousarray(frame[::-1])   # flip vertically
    return frame


def process_video(
    model,
    device,
    file_path,
    imgsz,
    num_classes,
    conf_thresh,
    map_id_to_name,
    output_path,
):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = min(cap.get(cv2.CAP_PROP_FPS), 30.0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (orig_w, orig_h),
    )

    transforms = T.Compose(
        [
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
        ]
    )

    ls_trackers = {
        id_class: Sort(
            {
                "max_age": 15,
                "min_hits": 0,
                "iou_threshold": 0.2,
                "overlap_threshold": 0.4,
            }
        )
        for id_class in range(num_classes)
    }
    print(f"Processing {file_path}")
    for frame_count in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        boxes = boxes[scores >= conf_thresh]
        labels = labels[scores >= conf_thresh]
        scores = scores[scores >= conf_thresh]

        for id_class in range(num_classes):
            alive_tracks, dead_tracks = ls_trackers[id_class].update(
                [
                    DetectionInfo(person_box=box.tolist() + [scr, cls_])
                    for box, scr, cls_ in zip(
                        boxes[labels == id_class].detach().cpu().numpy(),
                        scores[labels == id_class].detach().cpu().numpy(),
                        labels[labels == id_class].detach().cpu().numpy(),
                    )
                ],
                frame,
            )
            # print(boxes.shape, len(alive_tracks))
            render_tracks(frame, alive_tracks, id_class, map_id_to_name)

        cv2.imwrite(
            os.path.join(
                os.path.dirname(output_path) + "/image.jpg",
            ),
            frame,
        )

        # Write the frame
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as {output_path}")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=True, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    imgsz = args.imgsz
    num_classes = cfg.yaml_cfg["num_classes"]
    output_path = args.output_path
    conf_thresh = args.conf_thresh
    map_id_to_name = eval(args.map_id_to_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if the input file is an image or a video
    file_path = args.input
    process_video(
        model,
        device,
        file_path,
        imgsz,
        num_classes,
        conf_thresh,
        map_id_to_name,
        output_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-s", "--imgsz", type=int, default=640)
    parser.add_argument("--conf_thresh", type=float)
    parser.add_argument("--map_id_to_name", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    main(args)
