import torch
import torch.nn as nn

import cv2
import numpy as np

import mmcv
from model_wrappers import CO_DINO_5scale_9encdoer_lsj_r50_3x_coco
from model_wrappers.mmdet.core.visualization import imshow_det_bboxes

def show_result_pyplot(model,
                        img,
                        result,
                        palette='coco',
                        score_thr=0.3,
                        bbox_color=(72, 101, 241),
                        text_color=(72, 101, 241),
                        mask_color=None,
                        thickness=2,
                        font_size=13,
                        win_name='',
                        show=False,
                        wait_time=0,
                        out_file=None):
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
        
    bboxes = np.vstack(bbox_result)
    
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
        
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=model.CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(device="cuda", model_config="/workspace/EmbededFCM/models/CO-DETR/projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_3x_coco.py", model_checkpoint="/workspace/EmbededFCM/checkpoints/co_dino_5scale_9encoder_lsj_r50_3x_coco.pth")
    
    # load image
    img = cv2.imread("/workspace/EmbededFCM/models/CO-DETR/demo/demo.jpg")
    
    features, metas = model.input_to_features(img, device)
    
    out = model.features_to_output(features, metas, device)[0]
    
    show_result_pyplot(
        model.model,
        img,
        out,
        palette='coco',
        score_thr=0.3,
        out_file="/workspace/outputs/debug.jpg"
    )
    
    print()
    
if __name__ == "__main__":
    main()