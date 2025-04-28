from typing import Any, Dict, List

import torch.nn as nn
from torch import Tensor

class BaseWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        
    def input_to_features(self, x, device: str) -> Dict:
        """ Make input images to feature maps with backbone """
        raise NotImplementedError
    
    def features_to_output(self, x:Dict, device: str):
        """ Make feature maps from decoder to completed outputs """
        raise NotImplementedError