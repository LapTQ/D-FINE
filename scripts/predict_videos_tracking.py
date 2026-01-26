import subprocess
import concurrent.futures
import os
import glob

# Configuration
# Adjust these paths as needed
DFINE_ROOT = "/home/laptq/laptq-prj-46/submodules/D-FINE"
CONFIG_NAME = "dfine_hgnetv2_m--prj57-v1"
WEIGHTS_PATH = (
    "/home/laptq/laptq-prj-46/runs/prj57-v1/dfine_hgnetv2_m/train/best_stg2.pth"
)
BASE_INPUT_DIR = "/home/laptq/laptq-prj-46/data/prj57/datasets/uae/1283ab8b45d7cc2849e57b54cd5360b8/hi_quality/10/19/16"
BASE_OUTPUT_DIR = "/home/laptq/laptq-prj-46/outputs/predict_results/dfinem-960-nms-sort"
DEVICE = "cuda:1"
IMGSZ = 960
CONF_THRESH = 0.4
MAP_ID_TO_NAME = "{0: 'pot', 1: 'man'}"
MAX_WORKERS = 3


# List of video files to process

LS_SUBPATHF = [
    p[len(BASE_INPUT_DIR) + 1 :]
    for p in sorted(
        glob.glob(f"{BASE_INPUT_DIR}/*1760858819634_62251.mkv")
    )
]


def main():
    # Using ProcessPoolExecutor similar to Reference: submodules/D-FINE/scripts/train.py
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                subprocess.run,
                args=(
                    f"python tools/inference/torch_inf-sort.py "
                    f"-c configs/dfine/{CONFIG_NAME}.yml "
                    f"-r {WEIGHTS_PATH} "
                    f'--input "{os.path.join(BASE_INPUT_DIR, subpathf)}" '
                    f"--device {DEVICE} "
                    f"--imgsz {IMGSZ} "
                    f"--conf_thresh {CONF_THRESH} "
                    f'--map_id_to_name "{MAP_ID_TO_NAME}" '
                    f'--output_path "{os.path.join(BASE_OUTPUT_DIR, subpathf)}"'
                ),
                cwd=DFINE_ROOT,  # Execute from D-FINE root directory
                shell=True,
                check=True,
                text=True,
            )
            for subpathf in LS_SUBPATHF
        ]

        for i, f in enumerate(futures):
            try:
                f.result()
                print(f"[{i+1}/{len(LS_SUBPATHF)}] Completed: {LS_SUBPATHF[i]}")
            except subprocess.CalledProcessError as e:
                print(
                    f"[{i+1}/{len(LS_SUBPATHF)}] Failed: {LS_SUBPATHF[i]}\nError: {e}"
                )


if __name__ == "__main__":
    main()
