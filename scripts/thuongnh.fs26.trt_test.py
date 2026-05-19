"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import collections
import contextlib
import os
import time
from collections import OrderedDict

import cv2  # Added for video processing
import numpy as np
import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(
        self, engine_path, device="cuda:0", backend="torch", max_batch_size=8, verbose=False
    ):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()


# class TRTInference(object):
#     def __init__(self, engine_path, device="cuda:0", max_batch_size=8, verbose=False):
#         self.engine_path = engine_path
#         self.device = torch.device(device)
#         self.max_batch_size = max_batch_size

#         # 1. Logger & Runtime
#         self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        
#         # 2. Load Engine (TRT 10.x compatible)
#         self.engine = self.load_engine(engine_path)
        
#         # 3. Create Context
#         self.context = self.engine.create_execution_context()
        
#         # 4. Setup Tensors (Name-based instead of Binding-based)
#         self.input_names = []
#         self.output_names = []
#         self.tensors = OrderedDict() # Stores our torch buffers

#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             mode = self.engine.get_tensor_mode(name)
#             dtype = trt.nptype(self.engine.get_tensor_dtype(name))
#             shape = list(self.engine.get_tensor_shape(name))

#             if mode == trt.TensorIOMode.INPUT:
#                 self.input_names.append(name)
#                 # Handle dynamic batch size for inputs
#                 if shape[0] == -1:
#                     shape[0] = self.max_batch_size
#                     self.context.set_input_shape(name, shape)
#             else:
#                 self.output_names.append(name)

#             # Create torch buffer on the correct device
#             data = torch.empty(tuple(shape), dtype=torch.from_numpy(np.empty(0, dtype=dtype)).dtype, device=self.device)
#             self.tensors[name] = data
            
#             # CRITICAL for TRT 10.x: Set tensor address once
#             self.context.set_tensor_address(name, data.data_ptr())

#     def load_engine(self, path):
#         trt.init_libnvinfer_plugins(self.logger, "")
#         with open(path, "rb") as f:
#             # Note: No 'f.read(4)' here. If your file has metadata, use your custom loader.
#             engine_data = f.read()
#             with trt.Runtime(self.logger) as runtime:
#                 return runtime.deserialize_cuda_engine(engine_data)

#     def run(self, blob):
#         """
#         blob: dict of {name: torch.Tensor}
#         """
#         for name in self.input_names:
#             input_data = blob[name]
            
#             # 1. Update shape if dynamic
#             if tuple(self.context.get_tensor_shape(name)) != input_data.shape:
#                 self.context.set_input_shape(name, input_data.shape)
            
#             # 2. Copy data to input buffer (if not already there)
#             if input_data.data_ptr() != self.tensors[name].data_ptr():
#                 self.tensors[name].copy_(input_data)

#         # 3. Execute (Using v3 for Blackwell/Thor)
#         # execute_v3 is the modern way to run inference in TRT 10.x
#         success = self.context.execute_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
#         if not success:
#             raise RuntimeError("TensorRT execution failed!")

#         # 4. Return results (outputs are updated in-place in self.tensors)
#         return {n: self.tensors[n] for n in self.output_names}

#     def __call__(self, blob):
#         return self.run(blob)

def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

    return images


def process_image(m, file_path, device):
    conf = 0.3
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil)[None]

    blob = {
        "images": im_data.to(device),
        "orig_target_sizes": orig_size.to(device),
    }
    output = 'outputs/output_dfm'
    os.makedirs(output, exist_ok=True)
    img_path = file_path.split('/')[-1]
    fout = img_path.split('.')[0] + '.txt'
    f = open(os.path.join(output, fout), 'w')
    import time 
    t0 = time.time()
    output = m(blob)
    t1=time.time()
    print('FPS here:', 1/(t1-t0))
    # print(output["boxes"])
    # print(output["labels"])
    labels = output['labels']
    scores = output['scores']
    boxes = output['boxes']
    for i,im in enumerate([im_pil]):
        scr = scores[i]
        box = boxes[i][scr>conf]
        label = labels[i][scr>conf]
        scrs =scr[scr>conf]
        # print(label)
        for j, b in enumerate(box):
            f.write(f'{int(label[j])} {scrs[j]} {round(abs(b[0]).item())} {round(abs(b[1]).item())} {round(abs(b[2]).item())} {round(abs(b[3]).item())}' + '\n')
    # result_images = draw([im_pil], output["labels"], output["boxes"], output["scores"])
    # result_images[0].save("trt_result.jpg")
    # print("Image processing complete. Result saved as 'result.jpg'.")
    return (t1-t0)


def process_video(m, file_path, device):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("trt_result.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
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
        orig_size = torch.tensor([w, h])[None].to(device)

        im_data = transforms(frame_pil)[None]

        blob = {
            "images": im_data.to(device),
            "orig_target_sizes": orig_size.to(device),
        }
        import time 
        t0 = time.time()
        output = m(blob)
        t1=time.time()
        print('FPS here:', 1/(t1-t0))

        # Draw detections on the frame
        result_images = draw([frame_pil], output["labels"], output["boxes"], output["scores"])

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result_video.mp4'.")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-trt", "--trt", type=str, required=True)
    # parser.add_argument("-i", "--input", type=str, required=True)
    # parser.add_argument("-d", "--device", type=str, default="cuda:0")

    # args = parser.parse_args()

    class Args:
        trt = "/mnt/ssd1t/Users/laptq/D-FINE/best_dfine_m_PersonHand.engine"
        input = "/mnt/ssd1t/Users/thuongnh/rf-detr/test_sat_personhand/images"
        device = "cuda:0"
    args = Args()

    m = TRTInference(args.trt, device=args.device)

    file_path = args.input
    i = 0
    t = 0
    for img in os.listdir(file_path):
        i+=1
        t_p = process_image(m, os.path.join(file_path, img), args.device)
        t+=t_p
    print(i)
    print(t)
    print('FPS average here:', i/t)

    # if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
    #     # Process as image
    #     process_image(m, file_path, args.device)
    # else:
    #     # Process as video
    #     process_video(m, file_path, args.device)