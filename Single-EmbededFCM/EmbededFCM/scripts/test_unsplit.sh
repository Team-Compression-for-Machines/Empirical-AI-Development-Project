#! /bin/bash

python3 -m embeded_fcm.fcm.test_unsplit \
    --dataset /workspace/datasets/SFU_HW_Obj \
    --device cuda \
    --checkpoint /workspace/EmbededFCM/embeded_fcm/checkpoints/co_dino_5scale_9encoder_lsj_r50_3x_coco.pth \
    --config /workspace/EmbededFCM/models/CO-DETR/projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_3x_coco.py \
    --save_dir /workspace/outputs/unsplit