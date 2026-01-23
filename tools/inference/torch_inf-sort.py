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
COLORS = np.random.randint(0, 256, size=(32, 3))
def render_tracks_1(img, track_results):
    for track in track_results:
        track_id = int(track.track_id)
        x1_p,y1_p,x2_p,y2_p, score, _ = track.box
        color = COLORS[track_id % len(COLORS)].tolist()
        cv2.rectangle(img, (int(x1_p), int(y1_p)), (int(x2_p), int(y2_p)), color, thickness=2)
        cv2.putText(img=img, text=f'Pot {track_id}', org=(int(x1_p+3), int(y1_p-5)), thickness=2, color=color, 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7)

def render_tracks_2(img, track_results):
    for track in track_results:
        track_id = int(track.track_id)
        x1_p,y1_p,x2_p,y2_p, score, _ = track.box
        color = COLORS[track_id % len(COLORS)].tolist()
        cv2.rectangle(img, (int(x1_p), int(y1_p)), (int(x2_p), int(y2_p)), color, thickness=2)
        cv2.putText(img=img, text=f'Man {track_id}', org=(int(x1_p+3), int(y1_p-5)), thickness=2, color=color, 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7)

from sort import Sort
from detection_info import DetectionInfo

def process_video(model, device, file_path, imgsz, output_dir):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(output_dir, os.path.basename(file_path)), fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
        ]
    )

    tracker_1 = Sort(
        {
            "max_age": 15,
            "min_hits": 2,
            "iou_threshold": 0.2,
            "overlap_threshold": 0.4,
        }
    )
    tracker_2 = Sort(
        {
            "max_age": 15,
            "min_hits": 2,
            "iou_threshold": 0.2,
            "overlap_threshold": 0.4,
        }
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        boxes = boxes[scores >= 0.4]
        labels = labels[scores >= 0.4]
        scores = scores[scores >= 0.4]

        alive_tracks_1, dead_tracks_1 = tracker_1.update([DetectionInfo(person_box=box.tolist() + [scr, cls_]) for box, scr, cls_ in zip(boxes[labels==0].detach().cpu().numpy(), scores[labels==0].detach().cpu().numpy(), labels[labels==0].detach().cpu().numpy())], frame)
        alive_tracks_2, dead_tracks_2 = tracker_2.update([DetectionInfo(person_box=box.tolist() + [scr, cls_]) for box, scr, cls_ in zip(boxes[labels==1].detach().cpu().numpy(), scores[labels==1].detach().cpu().numpy(), labels[labels==1].detach().cpu().numpy())], frame)

        print(boxes.shape, len(alive_tracks_1), len(alive_tracks_2))

        render_tracks_1(frame, alive_tracks_1)
        render_tracks_2(frame, alive_tracks_2)

        cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.jpg'), frame)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


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
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Check if the input file is an image or a video
    file_path = args.input
    process_video(model, device, file_path, imgsz, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-s", "--imgsz", type=int, default=640)   
    parser.add_argument("--output_dir", type=str)    
    args = parser.parse_args()
    main(args)
