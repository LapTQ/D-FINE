# DEVICE=0
# PORT=7780
# NAME_CFG=dfine_hgnetv2_s--20241122--phase-2--annotation-ver2.yml

# cd submodules/D-FINE
# CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
#     --master_port=$PORT \
#     --nproc_per_node=1 \
#     train.py \
#     -c /mnt/hdd10tb/Users/laptq/laptq-prj-46/submodules/D-FINE/configs/dfine/$NAME_CFG \
#     --use-amp \
#     --seed=0 \

ls_config = [
    "dfine_hgnetv2_s--lagenda",
    "dfine_hgnetv2_s--virat",
    "dfine_hgnetv2_s--lagenda--virat",
]
ls_device = [0, 0, 0]

assert len(ls_config) == len(ls_device)

import subprocess
import multiprocessing as mp
import concurrent.futures

with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(
            subprocess.run,
            args=f"CUDA_VISIBLE_DEVICES={device} python3 train.py -c configs/dfine/{cfg}.yml --use-amp --seed=0 -t /home/pocuser2/laptq-nedo-fed/dfine_s_obj365.pth",
            cwd="submodules/D-FINE",
            shell=True,
            check=True,
            text=True,
        )
        for cfg, device in zip(ls_config, ls_device)
    ]
    ls_trained_model = [f.result() for f in futures]


# fuser -k -9 /dev/nvidia0