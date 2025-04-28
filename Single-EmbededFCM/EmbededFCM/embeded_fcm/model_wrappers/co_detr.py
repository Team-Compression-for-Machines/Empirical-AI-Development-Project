import torch
import time
from ptflops import get_model_complexity_info

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from .mmdet.apis import init_detector
from .mmdet.datasets import replace_ImageToTensor
from .mmdet.datasets.pipelines import Compose
from .mmdet.core import bbox2result

from .base_wrappers import BaseWrapper
from .projects import *

class CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(BaseWrapper):
    def __init__(self, device: str, model_config, model_checkpoint, **kwargs):
        super().__init__(device)
        
        self.model = init_detector(config=model_config, checkpoint=model_checkpoint, device=device)
        
        self.cfg = self.model.cfg
    
    def unsplit(self, x, device):
        nn_part_1_start_time = time.time()
        features, metas = self.input_to_features(x, device)
        nn_part_1_time = time.time() - nn_part_1_start_time
        
        nn_part_2_start_time = time.time()
        results = self.features_to_output(features, metas, device)[0]
        nn_part_2_time = time.time() - nn_part_2_start_time
        
        return {
            "nn_part_1_time": nn_part_1_time,
            "nn_part_2_time": nn_part_2_time,
            "results": results
        }
    
    def input_to_features(self, x, device):
        self.model = self.model.to(device).eval()
        
        return self._input_to_backbone(x, device)
    
    def features_to_output(self, x, metas, device):
        self.model = self.model.to(device).eval()
             
        results, x = self.model.query_head.simple_test(x, metas, rescale=True, return_encoder_output=True)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.model.query_head.num_classes)
            for det_bboxes, det_labels in results
        ]
        
        return bbox_results
    
    def _input_to_backbone(self, x, device):
        with torch.no_grad():
            self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            
            self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
            test_pipeline = Compose(self.cfg.data.test.pipeline)
            
            data = dict(img=x)
            data = test_pipeline(data)
            
            data['img_metas'] = [img_metas.data for img_metas in data['img_metas']]
            data['img'] = [img.data for img in data['img']]
            
            if next(self.model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                for m in self.model.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'
            
            data['img_metas'][0]['batch_input_shape'] = data['img'][0].size()[1:]
            return self.model.extract_feat(data['img'][0].unsqueeze(0)), data['img_metas']
        
    def profile_model(self, x, device):
        with torch.no_grad():
            self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            
            self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
            test_pipeline = Compose(self.cfg.data.test.pipeline)
            
            data = dict(img=x)
            data = test_pipeline(data)
            
            data['img_metas'] = [img_metas.data for img_metas in data['img_metas']]
            data['img'] = [img.data for img in data['img']]
            
            if next(self.model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                for m in self.model.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'
            
            data['img_metas'][0]['batch_input_shape'] = data['img'][0].size()[1:]
            
            macs, pixels, neck_results = self.profile_backbone(data['img'][0].unsqueeze(0))
            kmacs_per_pixels_nn_part_1 = macs / 1_000 / pixels
            
            # macs = self.profile_nn_part_2(neck_results, data['img_metas'], device)
            # kmacs_per_pixels += macs / 1_000
            # kmacs_per_pixels /= pixels
            
        return {
            "kmacs_per_pixels_nn_part_1": kmacs_per_pixels_nn_part_1
        }
    
    def profile_backbone(self, x):
        with torch.no_grad():
            B, C, H, W = x.shape
            pixels = B * C * H * W
            macs_sum = 0.0
            
            macs, _ = get_model_complexity_info(
                self.model.backbone,
                input_res=(C, H, W),
                input_constructor=None,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            macs_sum += macs
            
            backbone_results = self.model.backbone(x)
            input_res = tuple(tuple(f.shape) for f in backbone_results)
            
            macs, _ = get_model_complexity_info(
                self.model.neck,
                input_res=input_res,
                input_constructor=self.neck_input_constructors,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False 
            )
            macs_sum += macs
            
            neck_results = self.model.neck(backbone_results)
            
        return macs_sum, pixels, neck_results
    
    def profile_nn_part_2(self, features, metas, device):
        head = self.model.query_head.to(device).eval()
        wrapper = HeadProfilingWrappers(head, metas).to(device)
        
        with torch.no_grad():
            shapes = ([tuple(f.shape) for f in features], wrapper)
            
            macs, params = get_model_complexity_info(
                wrapper,
                input_res=shapes,
                input_constructor=self.head_input_constructor,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            
        return macs
    
    def neck_input_constructors(self, res_shapes):
        device = next(self.model.neck.parameters()).device
        
        feats = [torch.zeros(s, device=device) for s in res_shapes]
        return {"inputs": feats}
    
    def head_input_constructor(self, res_shapes):
        shape, wrapper = res_shapes
        dev = next(wrapper.parameters()).device
        dummy_feats = [torch.zeros(sz, device=dev) for sz in shape]
        return {
            "x": dummy_feats
        }
        
class HeadProfilingWrappers(nn.Module):
    def __init__(self, head, metas):
        super().__init__()
        self.head = head
        self.metas = metas
        
    def forward(self, x):
        return self.head.simple_test(x, self.metas, rescale=True, return_encoder_output=True)
        