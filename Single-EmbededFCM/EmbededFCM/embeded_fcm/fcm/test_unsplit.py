import argparse
import sys
import cv2
import numpy as np
import time
import psutil
import pynvml
from ptflops import get_model_complexity_info
import os
import json
from tqdm import tqdm
import logging
import contextlib
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mmcv
from embeded_fcm.model_wrappers.co_detr import CO_DINO_5scale_9encdoer_lsj_r50_3x_coco

from embeded_fcm.data.test_dataset import SFUHW

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Unsplit CO-DETR Inference")
    
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/workspace/datasets/SFU_HW_Obj"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint file path for use"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file path for use"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory path for save results"
    )
    
    args = parser.parse_args(argv)
    return args

def setup_logger():
    logger = logging.getLogger("unsplit_inference")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    return logger

def test(model, args, logger):
    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    model = model.to(device).eval()
    dataset_list = os.listdir(args.dataset)
    
    all_stats = {
        "total_start": time.time(),
        "datasets": []
    }
    
    for data in tqdm(dataset_list, desc="Datasets", leave=False):
        logger.info(f"Start processing: {data}")
        ds_start = time.time()
        
        dataset = SFUHW(
            root=os.path.join(args.dataset, data),
            annotation_file=f"annotations/{data}.json"
        )
        data_infos = dataset.load_annotations(dataset.annotation_path)
        dataset.data_infos = data_infos
        dataset._dataset = data_infos
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            pin_memory=(device == "cuda"),
            collate_fn=lambda x:x
        )
        
        kmac_per_pixels_total = 0.0
        
        results = []
        model_all_time = 0.0
        model_nn_part_1_time = 0.0
        model_nn_part_2_time = 0.0
        for idx, batch in tqdm(enumerate(dataloader), desc=f"DATA: {data}"):
            data_start_time = time.time()
            imgs = batch[0]['img']
            with torch.no_grad():
                result = model.unsplit(imgs, device)
            data_end_time = time.time() - data_start_time
            
            model_all_time += data_end_time
            model_nn_part_1_time += result["nn_part_1_time"]
            model_nn_part_2_time += result["nn_part_2_time"]
            
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                kmac_per_pixels = model.profile_model(imgs, device)
                kmac_per_pixels_total += kmac_per_pixels["kmacs_per_pixels_nn_part_1"]
            results.append(result["results"])
            
        eval_results = dataset.evaluate(
            results,
            metric='bbox'
        )
        
        ds_end = time.time()
        ds_time = ds_end - ds_start
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used_mb = mem_info.used / 1024 ** 2
        
        ram_used_mb = psutil.virtual_memory().used / 1024 ** 2
        
        stats = {
            "dataset": data,
            "time_sec": ds_time,
            "model_all_time": model_all_time,
            "model_part_1_time": model_nn_part_1_time,
            "model_part_2_time": model_nn_part_2_time,
            "gpu_used_mb": gpu_used_mb,
            "kmac_per_pixels": kmac_per_pixels_total,
            "eval_bbox": eval_results
        }
        all_stats["datasets"].append(stats)
        
        # print results and complexity on terminal
        logger.info(f"Completed: {data} | time: {ds_time:.2f}s | GPU: {gpu_used_mb:.2f}MB | RAM: {ram_used_mb:.2f}MB | kMAC/pixel: {kmac_per_pixels_total:.3f}")
    
    total_time = time.time() - all_stats["total_start"]
    all_stats["total_time"] = total_time
    
    # save results on json file
    out_file = os.path.join(args.save_dir, "inference_results.json")
    with open(out_file, "w") as f:
        json.dump(all_stats, f, indent=2)

def main(argv):
    args = parse_args(argv)
    device = args.device
    
    logger = setup_logger()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model_config = args.config
    model_checkpoint = args.checkpoint
    model = CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(
        device=device,
        model_config=model_config,
        model_checkpoint=model_checkpoint
    )
    
    test(
        model,
        args,
        logger
    )

if __name__ == "__main__":
    main(sys.argv[1:])